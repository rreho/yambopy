import numpy as np
try:
    from pykdtree.kdtree import KDTree 
    ## pykdtree is much faster and is recommanded
    ## pip install pykdtree
    ## useful in Dmat computation
except ImportError as e:
    from scipy.spatial import KDTree

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
    '''\bra{u_{v'k-Q}}\ket{u_{vk-Q}}'''
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

    for ik, _ in enumerate(kmq_grid):
        for iq, ikmq in enumerate(qs[ik]):
            # kmq = nnkp_kgrid.k[ikmq]
            G0_bra = [0,0,0] # We are already using wannier90 grid
            G0_ket = [0,0,0] # We are already using wannier90 grid
            wfc_k1, gvec_k1 = wfdb.get_BZ_wf(qs[ik,iq])
            # wfc_k2, gvec_k2 = wfdb.get_BZ_wf(qs[ik,iq])

            # wfc_k2, gvec_k2 = wfdb.get_BZ_wf(iq)
            Mkmq[ik,iq] = Mmn_kkp(G0_bra, wfc_k1, gvec_k1, G0_ket, wfc_k1, gvec_k1)
    return Mkmq

def compute_overlap_kkpb(wfdb, nnkp):
    '''\bra{u_{ck}}\ket{u_{c'k+ B}}'''
    if getattr(wfdb, 'wf_bz', None) is None: wfdb.expand_fullBZ()
    nk = wfdb.nkpoints
    nb = 8
    k_bra = nnkp.data[:,0]-1
    k_ket = nnkp.data[:,1]-1
    Gs_ket = nnkp.data[:,2:]

    nbands = wfdb.nbands

    Mkpb = np.zeros(shape=(nk,nb,nbands,nbands),dtype=np.complex128)

    for ik, k1 in enumerate(k_bra):
        
        k2 = k_ket[ik]
        ib = ik% nb

        G0_bra = [0,0,0]
        G0_ket = Gs_ket[ik]
        print(int(k1))
        wfc_k1, gvec_k1 = wfdb.get_BZ_wf(int(k1))
        wfc_k2, gvec_k2 = wfdb.get_BZ_wf(int(k2))
            
        Mkpb[int(ik/nb),ib] = Mmn_kkp(G0_bra, wfc_k1, gvec_k1,G0_ket, wfc_k2, gvec_k2)
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

def Mmn_kkp(G0_bra, wfc_bra, gvec_bra, G0_ket, wfc_ket, gvec_ket, ket_Gtree=None):
    """
    Computes the inner product between two wavefunctions in reciprocal space. <k_bra | k_ket>
    
    Parameters
    ----------
    k_bra : ndarray
        Crystal momentum of the bra wavefunction (3,) in reduced coordinates.
    wfc_bra : ndarray
        Wavefunction coefficients for the bra state with shape (nspin, nbnd, nspinor, ng) of yambo.
    gvec_bra : ndarray
        Miller indices of the bra wavefunction (ng, 3) in reduced coordinates of yambo.
    wfc_ket : ndarray
        Wavefunction coefficients for k+b the ket state with shape (nspin, nbnd, nspinor, ng) of yambo.
    gvec_ket : ndarray
        Miller indices of the ket wavefunction (ng, 3) in reduced coordinates of yambo.
    ket_Gtree  : scipy.spatial._kdtree.KDTree (optional)
        Kdtree for gvec_ket. leave it or give None to internally build one
    G0 = k_w - ky # difference between wannier and yambo k vectors
    #
    Returns
    -------
    ndarray
        Inner product matrix of shape (nspin, nbnd, nbnd). If the momenta mismatch
        is too large, returns a zero matrix.
    """
    #
    # Check consistency of wavefunction dimensions
    assert wfc_ket.shape[:3] == wfc_bra.shape[:3], "Inconsistant wfcs"
    #
    nspin, nbnd, nspinor = wfc_ket.shape[:3]

    # Construct KDTree for nearest-neighbor search in G-vectors
    if ket_Gtree is None:
        ket_Gtree = KDTree(gvec_ket-G0_ket)
    gbra_shift = gvec_bra - G0_bra#+ G0[None,:]
    ## get the nearest indices and their distance
    dd, ii = ket_Gtree.query(gbra_shift, k=1)
    #
    wfc_bra_tmp = np.zeros(wfc_ket.shape,dtype=wfc_ket.dtype)
    # Get only the indices that are present
    bra_idx = ii[dd < 1e-6]
    #
    wfc_bra_tmp[:,:,:,bra_idx] = wfc_bra[...,dd<1e-6].conj()
    # return the dot product
    inprod = np.zeros((nspin, nbnd, nbnd),dtype=wfc_bra.dtype)
    for ispin in range(nspin):
        inprod[ispin] = wfc_bra_tmp[ispin].reshape(nbnd,-1)@wfc_ket[ispin].reshape(nbnd,-1).T
    #return np.einsum('sixg,sjxg->sij',wfc_bra_tmp,wfc_ket,optimize=True) #// einsum is very slow
    return inprod