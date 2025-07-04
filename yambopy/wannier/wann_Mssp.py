import numpy as np


def convert_to_wannier90(wfdb, nnkp_kgrid):
    "WFdb conversion to wannier90 kgrid"
    y2w = nnkp_kgrid.yambotowannier90_table

    wfdb.kBZ = wfdb.kBZ[y2w]
    wfdb.wf_bz = wfdb.wf_bz[y2w]
    wfdb.gvecs = wfdb.gvecs[y2w]
    wfdb.ngBZ = wfdb.ngBZ[y2w]
    wfdb.wannier90 = True
    print("Converted to Wannier90 grids.")

def compute_overlap_kmq(wfdb, nnkp_kgrid):
    from yambopy.dbs.wfdb import wfc_inner_product
    if getattr(wfdb, 'wf_bz', None) is None: wfdb.expand_fullBZ()
    if getattr(wfdb, 'wannier90', None) is None:convert_to_wannier90(wfdb, nnkp_kgrid)
    kmq_table = nnkp_kgrid.kmq_grid_table
    kmq_grid = nnkp_kgrid.kmq_grid
    ks = kmq_table[:,:,0]
    qs = kmq_table[:,:,1]
    nk = len(ks)
    nq = len(qs)
    nbands = wfdb.nbands

    Mkmq = np.zeros(shape=(nk,nq,nbands,nbands),dtype=np.complex128)

    for ik, kmq_1 in enumerate(kmq_grid):
        for iq, kmq_2 in enumerate(kmq_1):
            wfc_k1, gvec_k1 = wfdb.get_BZ_wf(qs[ik,iq])
            wfc_k2, gvec_k2 = wfdb.get_BZ_wf(qs[ik,iq])

            # wfc_k2, gvec_k2 = wfdb.get_BZ_wf(iq)
            Mkmq[ik,iq] = wfc_inner_product(kmq_2, wfc_k1, gvec_k1, kmq_2, wfc_k1, gvec_k1)
    return Mkmq

def compute_overlap_kkpb(wfdb, nnkp_kgrid):
    from yambopy.dbs.wfdb import wfc_inner_product
    if getattr(wfdb, 'wf_bz', None) is None: wfdb.expand_fullBZ()
    if getattr(wfdb, 'wannier90', None) is None:convert_to_wannier90(wfdb, nnkp_kgrid)
    kpb_table = nnkp_kgrid.kpb_grid_table
    kpb_grid = nnkp_kgrid.kpb_grid
    ks = kpb_table[:,:,0]
    bs = kpb_table[:,:,1]
    nk = len(ks)
    nb = kpb_grid.shape[1]
    nbands = wfdb.nbands

    Mkpb = np.zeros(shape=(nk,nb,nbands,nbands),dtype=np.complex128)

    for ik, k in enumerate(ks):
        for ib, kpb in enumerate(bs[ik]):
            k1 = nnkp_kgrid.k[ik]
            k2 = kpb_grid[ik,ib]
            wfc_k1, gvec_k1 = wfdb.get_BZ_wf(ik)
            wfc_k2, gvec_k2 = wfdb.get_BZ_wf(kpb)
            # wfc_k2, gvec_k2 = wfdb.get_BZ_wf(bs[ik,ib])

            # wfc_k2, gvec_k2 = wfdb.get_BZ_wf(iq)
            Mkpb[ik,ib] = wfc_inner_product(k1, wfc_k1, gvec_k1, k2, wfc_k1, gvec_k1)
    return Mkpb



def compute_Mssp(h2p,nnkp_kgrid,nnkp_qgrid,trange=1):
    nb = nnkp_kgrid.b_list[0].shape[0]
    Mssp = np.zeros(shape=(trange,trange,h2p.nq,h2p.nb ))
    for t in range(0,trange):
        for tp in range(0,trange):
            for iq, q in enumerate(nnkp_qgrid.red_kpoints):
                print(iq)
                for ib in range(nb):
                    Mssp_ttp = 0
                    iqpb = h2p.qmpgrid.qpb_grid_table[iq, ib, 1]
                    bset = h2p.bse_nc*h2p.bse_nv
                    k = h2p.BSE_table[:, 0]
                    v = h2p.BSE_table[:, 1]
                    c = h2p.BSE_table[:, 2]
                    for ik, iv, ic in zip(k,v,c):  # ∑_{cvk}
                        ikpb = h2p.kmpgrid.kpb_grid_table[ik, ib, 1]  # (N, 1)
                        ikmq = h2p.kmpgrid.kmq_grid_table[ik, iq, 1]  # (N, 1)
                        for ivp, icp in zip(v[:bset], c[:bset]):

                            # term1: A^{SQ*}_{cvk}
                            term1 = np.conjugate(h2p.h2peigvec_vck[iq, t, h2p.bse_nv - h2p.nv + iv, ic - h2p.nv, ik])  # shape (N, 1)
                            # term2: A^{S'Q+B}_{c'v'k+B}
                            term2 = h2p.h2peigvec_vck[iqpb, tp, h2p.bse_nv - h2p.nv + ivp, icp - h2p.nv, ikpb]  # shape (N, M)
                            term3 = h2p.Mkpb[ik,ib,ic-1, icp-1] # this is already saved in terms of kpb
                            term4 = h2p.Mkmq[ik,iq,ivp-1, iv-1] # This is already saved in terms of ikmq
                            
                            Mssp_ttp += np.sum(term1 * term2 * term3 * term4)  # scalar

                    Mssp[t,tp,iq,ib] = Mssp_ttp
    h2p.Mssp = Mssp
    return Mssp
