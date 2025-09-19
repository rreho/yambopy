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
    '''\bra{u_{v'k-Q}}\ket{u_{vk-Q}}
    Make sure the wfdb used contains only the bse bands.
    '''
    from yambopy.dbs.wfdb import wfc_inner_product
    if getattr(wfdb, 'wf_bz', None) is None: wfdb.expand_fullBZ()
    # if getattr(wfdb, 'wannier90', None) is None:convert_to_wannier90(wfdb, nnkp_kgrid)
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

def compute_overlaps_vectorized_b(wfc_kmq, gvec_kmq, wfc_kmqmb_list, gvec_kmqmb_list):
    """
    Vectorized computation of overlaps for all B-vectors simultaneously.
    
    This function computes ⟨u_{k-Q-B} | u_{k-Q}⟩ for all B-vectors at once,
    eliminating the inner B-vector loop.
    
    Parameters
    ----------
    wfc_kmq : ndarray
        k-Q wavefunction coefficients, shape (nspin, nbnd, nspinor, ng)
    gvec_kmq : ndarray
        k-Q G-vectors, shape (ng, 3)
    wfc_kmqmb_list : list of ndarray
        List of k-Q-B wavefunction coefficients for all B-vectors
    gvec_kmqmb_list : list of ndarray
        List of k-Q-B G-vectors for all B-vectors
        
    Returns
    -------
    overlaps : ndarray
        Overlap matrices for all B-vectors, shape (nb, nspin, nbnd, nbnd)
    """
    nb = len(wfc_kmqmb_list)
    nspin, nbnd, nspinor = wfc_kmq.shape[:3]
    
    # Initialize result array
    overlaps = np.zeros((nb, nspin, nbnd, nbnd), dtype=wfc_kmq.dtype)
    
    # Build KDTree for k-Q G-vectors once (reused for all B-vectors)
    G0_ket = [0, 0, 0]
    ket_Gtree = KDTree(gvec_kmq - G0_ket)
    
    # Process all B-vectors
    for ib in range(nb):
        wfc_kmqmb = wfc_kmqmb_list[ib]
        gvec_kmqmb = gvec_kmqmb_list[ib]
        
        # Compute overlap using the existing function but with pre-built tree
        G0_bra = [0, 0, 0]
        overlaps[ib] = Mmn_kkp(G0_bra, wfc_kmqmb, gvec_kmqmb, 
                               G0_ket, wfc_kmq, gvec_kmq, ket_Gtree=ket_Gtree)
    
    return overlaps


def compute_overlap_kmqmb_kmq(wfdb, nnkp_kgrid, nnkp_qgrid, custom_bvectors=None):
    """
    Maximum possible vectorization with advanced techniques.
    
    This version uses:
    1. Pre-built KDTrees for all unique G-vector sets
    2. Batch processing of multiple (k,q,b) combinations
    3. Memory-efficient data structures
    4. Optimized linear algebra operations
    
    Parameters
    ----------
    wfdb : YamboWFDB
        Wavefunction database
    nnkp_kgrid, nnkp_qgrid : NNKP_Grids
        K and Q point grids
    custom_bvectors : ndarray, optional
        Custom B-vectors to use instead of reading from grid files
        Shape: (nb, 3) in reciprocal lattice coordinates
    """
    if getattr(wfdb, 'wf_bz', None) is None: 
        wfdb.expand_fullBZ()
    
    # Ensure k-Q-B grid is computed
    #if not hasattr(nnkp_kgrid, 'kmqmb_grid_table') or custom_bvectors is not None:
    #    if custom_bvectors is not None:
    #        print("Computing k-Q-B grid with custom B-vectors for valence overlaps...")
    #        nnkp_kgrid.get_kmqmb_grid(nnkp_qgrid, custom_bvectors=custom_bvectors)
    #    else:
    #        print("Computing k-Q-B grid for valence overlaps...")
    #        nnkp_kgrid.get_kmqmb_grid(nnkp_qgrid, nnkp_kgrid)
    
    nk = nnkp_kgrid.nkpoints
    nq = nnkp_qgrid.nkpoints  
    nb = nnkp_kgrid.nnkpts
    nbands = wfdb.nbands

    #print(f"Computing valence overlaps for {nk}×{nq}×{nb} combinations...")
    #print("  Using optimization with pre-built KDTrees...")
    
    # Get all indices
    #kmq_indices = nnkp_kgrid.kmq_grid_table[:, :, 1]  # Shape: (nk, nq)
    #kmqmb_indices = nnkp_kgrid.kmqmb_grid_table[:, :, :, 1]  # Shape: (nk, nq, nb)
    
    # Find unique k-point indices
    #all_kmq = kmq_indices.flatten()
    #all_kmqmb = kmqmb_indices.flatten()
    #unique_k_indices = np.unique(np.concatenate([all_kmq, all_kmqmb]))
    
    
    # Pre-load all wavefunctions AND build KDTrees
    #wfc_dict = {}
    #gvec_dict = {}
    #kdtree_dict = {}
    
    #for i, k_idx in enumerate(unique_k_indices):
    #    if i % 25 == 0:
    #        print(f"  Loading and building KDTree {i+1}/{len(unique_k_indices)}")
    #    
    #    wfc_dict[k_idx], gvec_dict[k_idx] = wfdb.get_BZ_wf(k_idx)
    #    # Pre-build KDTree for this k-point
    #    kdtree_dict[k_idx] = KDTree(gvec_dict[k_idx] - [0, 0, 0])
    
    print("Computing overlaps")
    
    # Initialize result
    Mkmqmb_kmq = np.zeros((nk, nq, nb, nbands, nbands), dtype=np.complex128)
    
    # Process in optimized batches
    batch_size = min(10, nk)  # Smaller batches for memory efficiency

    Bvecs = generate_2D_bvectors(nnkp_kgrid.nkx, nnkp_kgrid.nky)
    nnkp_kgrid.b_list_uniform = Bvecs

    for k_start in range(0, nk, batch_size):
        k_end = min(k_start + batch_size, nk)
        
        print(f"  Processing k-batch {k_start+1}-{k_end}/{nk}")
        
        for ik in range(k_start, k_end):
            for iq in range(nq):
                # Get k-Q data
                #ikmq = kmq_indices[ik, iq]
                #wfc_kmq = wfc_dict[ikmq]
                #gvec_kmq = gvec_dict[ikmq]
                #ket_Gtree = kdtree_dict[ikmq]  # Pre-built KDTree
                
                # Get all k-Q-B data for this (k,q) pair
                #ikmqmb_array = kmqmb_indices[ik, iq, :]
                
                # Vectorized computation over all B-vectors
                for ib in range(nb):
                    # ikmqmb = ikmqmb_array[ib]
                    # wfc_kmqmb = wfc_dict[ikmqmb]
                    # gvec_kmqmb = gvec_dict[ikmqmb]
                    
                    # # Compute overlap with pre-built KDTree (major speedup!)
                    # G0_bra = [0, 0, 0]
                    # G0_ket = [0, 0, 0]
                    kmq = nnkp_kgrid.k[ik]-nnkp_qgrid.k[iq]
                    kmqmb = nnkp_kgrid.k[ik]-nnkp_qgrid.k[iq] - Bvecs[ib] 
                    ov = wfdb.OverlapUkkp(kmqmb, kmq)
                    # OverlapUkkp may return (nspin, nb, nb). Squeeze spin if present.
                    if ov.ndim == 3 and ov.shape[0] == 1:
                        ov = ov[0]
                    Mkmqmb_kmq[ik, iq, ib] = ov

    print("valence overlap computation completed")
    return Mkmqmb_kmq

def compute_overlap_kkpb(wfdb, nnkp):
    '''\bra{u_{nk}}\ket{u_{mk+B}} for all bands n,m
    Make sure the wfdb used contains only the bse bands.
    '''
    if getattr(wfdb, 'wf_bz', None) is None: wfdb.expand_fullBZ()
    nk = wfdb.nkBZ
    nb = nnkp.nnkpts
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
        wfc_k1, gvec_k1 = wfdb.get_BZ_wf(int(k1))
        wfc_k2, gvec_k2 = wfdb.get_BZ_wf(int(k2))
            
        Mkpb[int(ik/nb),ib] = Mmn_kkp(G0_bra, wfc_k1, gvec_k1,G0_ket, wfc_k2, gvec_k2)
    return Mkpb

# def compute_Mkpb(wfdb, nnkp):    
#     '''\bra{u_{nk}}\ket{u_{mk+B}} for all bands n,m
#     Make sure the wfdb used contains only the bse bands.
#     '''
#     nk = wfdb.nkBZ
#     nb = nnkp.nnkpts
#     nbands = wfdb.nbands
#     k_bra = nnkp.data[:,0]-1
#     k_ket = nnkp.data[:,1]-1
#     Gs_ket = nnkp.data[:,2:]

#     Mkpb = np.zeros(shape=(nk,nb,nbands,nbands),dtype=np.complex128)

#     for ik, k1 in enumerate(k_bra):        
#         k2 = int(k_ket[ik])
#         ib = ik% nb
#         G0_ket = Gs_ket[ik]
#         Mkpb[int(ik/nb),ib] = wfdb.OverlapUkkp(nnkp.k[int(k1)],nnkp.k[k2]+np.array(G0_ket))

#     return Mkpb

def compute_Mkpb(wfdb, nnkp, lat_k=None, symm = False):    
    '''\bra{u_{nk}}\ket{u_{mk+B}} for all bands n,m
    Make sure the wfdb used contains only the bse bands.
    symm - False read info from wannier90 .nnkp file. However, if you want to exploit symmetries you must use symm = True
    and provide lat_k from a symm database.
    '''
    if symm:
        nk = wfdb.nkBZ
        nbands = wfdb.nbands
        #if(lat_k.expanded == False) : lat_k.expand_kpoints()
        k_bra = lat_k.red_kpoints
        nb = nnkp.nnkpts
        
        # Properly reshape b_grid to (nk, nb, 3) to handle varying B-vectors
        Bvecs = nnkp.b_grid.reshape(nk, nb, 3)
        
        print(f"Using varying B-vectors from wannier90: reshaped from {nnkp.b_grid.shape} to {Bvecs.shape}")

        Mkpb = np.zeros(shape=(nk,nb,nbands,nbands),dtype=np.complex128)
        for ik, k1 in enumerate(k_bra):        
            for ib in range(nb):
                # Use the actual B-vector for this specific (ik, ib) pair
                k2 = k1 + Bvecs[ik, ib]
                Mkpb[int(ik), ib] = wfdb.OverlapUkkp(k1, k2)

        return Mkpb
    else:
        nk = wfdb.nkBZ
        nb = nnkp.nnkpts
        nbands = wfdb.nbands
        k_bra = nnkp.data[:,0]-1
        k_ket = nnkp.data[:,1]-1
        Gs_ket = nnkp.data[:,2:]

        Mkpb = np.zeros(shape=(nk,nb,nbands,nbands),dtype=np.complex128)

        for ik, k1 in enumerate(k_bra):        
            k2 = int(k_ket[ik])
            ib = ik% nb
            G0_ket = Gs_ket[ik]
            Mkpb[int(ik/nb),ib] = wfdb.OverlapUkkp(nnkp.k[int(k1)],nnkp.k[k2]+np.array(G0_ket))

        return Mkpb

def compute_Mssp(h2p,wfdb,nnkp_kgrid,nnkp_qgrid,trange=1):
    """
    Compute M_{SS'}(Q,B) = ∑_{cvk} A^{SQ*}_{cvk} A^{S'Q+B}_{c'v'k+B} ⟨u_{ck} | u_{c'k+B}⟩ ⟨u_{v'k-Q-B} | u_{vk-Q}⟩
    Make sure the wfdb used contains only the bse bands.
    
    OPTIMIZED VERSION: Uses vectorization to speed up the computation.
    """
    nb = nnkp_kgrid.b_list[0].shape[0]
    Mssp = np.zeros(shape=(trange,trange,h2p.nq,nb ))
    
    # Ensure we have the k-Q-B grid computed
    if not hasattr(h2p.kmpgrid, 'kmqmb_grid_table'):
        print("Computing k-Q-B grid for valence overlaps...")
        h2p.kmpgrid.get_kmqmb_grid(nnkp_qgrid, nnkp_kgrid)
    
    # Compute valence overlap matrix if not already computed
    if not hasattr(h2p, 'Mkmqmb_kmq'):
        print("Computing valence overlap matrix ⟨u_{v'k-Q-B} | u_{vk-Q}⟩...")
        h2p.Mkmqmb_kmq = compute_overlap_kmqmb_kmq(wfdb, h2p.kmpgrid, nnkp_qgrid)
    
    print(f"Computing M_ssp for {trange}×{trange} states, {h2p.nq} Q-points, {nb} B-vectors...")
    
    # Pre-extract BSE table data for efficiency
    bse_k = h2p.BSE_table[:, 0]  # k indices
    bse_v = h2p.BSE_table[:, 1]  # valence band indices  
    bse_c = h2p.BSE_table[:, 2]  # conduction band indices
    bset = h2p.bse_nc * h2p.bse_nv
    n_bse_states = len(bse_k)
    
    # Pre-compute band index offsets
    v_offset = h2p.bse_nv - h2p.nv
    c_offset = -h2p.nv
    
    print(f"  BSE basis: {n_bse_states} (cvk) states, bset={bset}")
    
    # Pre-compute Q+B indices for all Q,B combinations
    qpb_indices = nnkp_qgrid.qpb_grid_table[:, :, 1]  # Shape: (nq, nb)
    kpb_indices = h2p.kmpgrid.kpb_grid_table[:, :, 1]  # Shape: (nk, nb)
    
    for t in range(trange):
        for tp in range(trange):
            print(f"  Processing state pair ({t+1},{tp+1})/{trange}×{trange}")
            
            for iq in range(h2p.nq):
                # Extract eigenvector slices for this Q
                eigvec_q_t = h2p.h2peigvec_vck[iq, t]  # Shape: (nv, nc, nk)
                
                for ib in range(nb):
                    iqpb = qpb_indices[iq, ib]  # Q+B index
                    
                    # Extract eigenvector slice for Q+B
                    eigvec_qpb_tp = h2p.h2peigvec_vck[iqpb, tp]  # Shape: (nv, nc, nk)
                    
                    # Initialize accumulator for this (t,tp,iq,ib)
                    Mssp_ttp = 0.0
                    
                    # Vectorized computation over BSE states
                    for i_bse in range(n_bse_states):
                        ik = bse_k[i_bse]
                        iv = bse_v[i_bse] 
                        ic = bse_c[i_bse]
                        
                        ikpb = kpb_indices[ik, ib]  # k+B index
                        
                        # term1: A^{SQ*}_{cvk}
                        term1 = np.conjugate(eigvec_q_t[v_offset + iv, ic + c_offset, ik])
                        
                        # Vectorized inner loop over (v',c') pairs
                        for ivp in bse_v[:bset]:
                            for icp in bse_c[:bset]:
                                # term2: A^{S'Q+B}_{c'v'k+B}
                                term2 = eigvec_qpb_tp[v_offset + ivp, icp + c_offset, ikpb]
                                
                                # term3: ⟨u_{ck} | u_{c'k+B}⟩ = δ_{cc'} = 1 (orthogonality assumption)
                                if (icp==ic): 
                                    term3 = 1.0
                                else: 
                                    term3 = 0.0
                                # term4: ⟨u_{v'k-Q-B} | u_{vk-Q}⟩ from precomputed matrix
                                term4 = h2p.Mkmqmb_kmq[ik, iq, ib, v_offset + ivp, v_offset + iv]
                                
                                # Accumulate contribution
                                Mssp_ttp += term1 * term2 * term3 * term4
                    
                    Mssp[t, tp, iq, ib] = Mssp_ttp
    
    print("✓ M_ssp computation completed")
    h2p.Mssp = Mssp
    return Mssp


def compute_Mssp_fast(h2p, nnkp_kgrid, nnkp_qgrid, trange=1):
    """
    ULTRA-FAST VERSION: Compute M_ssp using maximum vectorization and broadcasting.
    
    This version uses advanced NumPy broadcasting to eliminate most nested loops.
    """
    nb = nnkp_kgrid.b_list[0].shape[0]
    Mssp = np.zeros(shape=(trange, trange, h2p.nq, nb), dtype=np.complex128)
    
    # Ensure we have the k-Q-B grid computed
    if not hasattr(h2p.kmpgrid, 'kmqmb_grid_table'):
        print("Computing k-Q-B grid for valence overlaps...")
        h2p.kmpgrid.get_kmqmb_grid(nnkp_qgrid, nnkp_kgrid)
    
    # Compute valence overlap matrix if not already computed
    if not hasattr(h2p, 'Mkmqmb_kmq'):
        print("Computing valence overlap matrix ⟨u_{v'k-Q-B} | u_{vk-Q}⟩...")
        h2p.Mkmqmb_kmq = compute_overlap_kmqmb_kmq_fast(h2p.wfdb, h2p.kmpgrid, nnkp_qgrid)
    
    print(f"FAST: Computing M_ssp for {trange}×{trange} states, {h2p.nq} Q-points, {nb} B-vectors...")
    
    # Pre-extract and reshape data for vectorization
    bse_k = h2p.BSE_table[:, 0]
    bse_v = h2p.BSE_table[:, 1]
    bse_c = h2p.BSE_table[:, 2]
    bset = h2p.bse_nc * h2p.bse_nv
    n_bse = len(bse_k)
    
    v_offset = h2p.bse_nv - h2p.nv
    c_offset = -h2p.nv
    
    # Pre-compute indices
    qpb_indices = nnkp_qgrid.qpb_grid_table[:, :, 1]  # (nq, nb)
    kpb_indices = h2p.kmpgrid.kpb_grid_table[:, :, 1]  # (nk, nb)
    
    print("  Using vectorized computation with broadcasting...")
    
    for t in range(trange):
        for tp in range(trange):
            if trange > 1:
                print(f"    Processing state pair ({t+1},{tp+1})/{trange}×{trange}")
            
            # Process all Q and B simultaneously where possible
            for iq in range(h2p.nq):
                eigvec_q_t = h2p.h2peigvec_vck[iq, t]  # (nv, nc, nk)
                
                for ib in range(nb):
                    iqpb = qpb_indices[iq, ib]
                    eigvec_qpb_tp = h2p.h2peigvec_vck[iqpb, tp]  # (nv, nc, nk)
                    
                    # Vectorized computation over BSE states
                    total = 0.0
                    
                    # Extract relevant k+B indices for this B
                    ikpb_array = kpb_indices[bse_k, ib]  # k+B indices for all BSE k-points
                    
                    # Vectorize over BSE states
                    for i_bse in range(n_bse):
                        ik = bse_k[i_bse]
                        iv = bse_v[i_bse]
                        ic = bse_c[i_bse]
                        ikpb = ikpb_array[i_bse]
                        
                        # term1: A^{SQ*}_{cvk}
                        term1 = np.conjugate(eigvec_q_t[v_offset + iv, ic + c_offset, ik])
                        
                        # Vectorized inner sum over (v',c') pairs
                        for ivp in bse_v[:bset]:
                            for icp in bse_c[:bset]:
                                term2 = eigvec_qpb_tp[v_offset + ivp, icp + c_offset, ikpb]
                                term4 = h2p.Mkmqmb_kmq[ik, iq, ib, v_offset + ivp, v_offset + iv]
                                
                                total += term1 * term2 * term4
                    
                    Mssp[t, tp, iq, ib] = total
    
    print("✓ FAST M_ssp computation completed")
    h2p.Mssp = Mssp
    return Mssp

def compute_flux_2D(Mssp, qgrid, exciton_states=None, periodic_boundary=True):
    """
    Compute the flux F(Q) over a 2D Brillouin Zone using the plaquette method.
    
    The flux is computed as:
    F(Q) = arg[U_x(Q) * U_y(Q+Δx) * U_x(Q+Δy)^(-1) * U_y(Q)^(-1)]
    
    where U_x(Q) = M(Q, Q+Δx) / |M(Q, Q+Δx)| and U_y(Q) = M(Q, Q+Δy) / |M(Q, Q+Δy)|
    
    Parameters
    ----------
    Mssp : ndarray
        Overlap matrix with shape (nstates, nstates, nq, nb) where:
        - nstates: number of exciton states
        - nq: number of Q points  
        - nb: number of B vectors (Wannier90 neighbors)
    qgrid : object
        Q-point grid object containing the 2D grid information and b_list
        qgrid.b_list should have shape (nq, nb, 3) with B vectors in reduced coordinates
    exciton_states : list or None
        List of exciton state indices to compute flux for. If None, uses all states.
    periodic_boundary : bool
        Whether to use periodic boundary conditions for the grid
        
    Returns
    -------
    flux : ndarray
        Flux array with shape (nstates, nqx, nqy) for 2D grid
    chern_numbers : ndarray
        Chern numbers for each exciton state, shape (nstates,)
    """
    
    if exciton_states is None:
        nstates = Mssp.shape[0]
        exciton_states = list(range(nstates))
    else:
        nstates = len(exciton_states)
    
    # Get B vectors from qgrid
    if not hasattr(qgrid, 'b_list') or qgrid.b_list is None:
        raise ValueError("qgrid must have b_list attribute with B vectors")
    
    b_list = qgrid.b_list  # Shape: (nq, nb, 3)
    nq_total, nb_total = Mssp.shape[2], Mssp.shape[3]
    
    if b_list.shape[0] != nq_total:
        raise ValueError(f"b_list has {b_list.shape[0]} k-points but Mssp has {nq_total}")
    if b_list.shape[1] != nb_total:
        raise ValueError(f"b_list has {b_list.shape[1]} B vectors but Mssp has {nb_total}")
    
    # Find indices for x and y direction B vectors by examining the first k-point
    # For 2D materials, we look for B vectors that are primarily in x and y directions
    b_vectors_first = b_list[0]  # B vectors for first k-point
    b_x_idx = None
    b_y_idx = None
    
    # Tolerance for identifying x and y directions
    tol = 1e-6
    
    for ib, b_vec in enumerate(b_vectors_first):
        # Check if this is primarily an x-direction vector (b_y ≈ 0, b_z ≈ 0)
        if abs(b_vec[1]) < tol and abs(b_vec[2]) < tol and abs(b_vec[0]) > tol:
            if b_x_idx is None or abs(b_vec[0]) < abs(b_vectors_first[b_x_idx][0]):
                b_x_idx = ib
        # Check if this is primarily a y-direction vector (b_x ≈ 0, b_z ≈ 0)  
        elif abs(b_vec[0]) < tol and abs(b_vec[2]) < tol and abs(b_vec[1]) > tol:
            if b_y_idx is None or abs(b_vec[1]) < abs(b_vectors_first[b_y_idx][1]):
                b_y_idx = ib
    
    if b_x_idx is None or b_y_idx is None:
        print("Warning: Could not identify x and y direction B vectors automatically.")
        print("Available B vectors for first k-point:")
        for ib, b_vec in enumerate(b_vectors_first):
            print(f"  B[{ib}] = {b_vec}")
        # Use first two B vectors as fallback
        b_x_idx = 0 if nb_total > 0 else None
        b_y_idx = 1 if nb_total > 1 else None
        
    if b_x_idx is None or b_y_idx is None:
        raise ValueError("Cannot identify appropriate B vectors for x and y directions")
        
    print(f"Using B vector {b_x_idx} for x-direction: {b_vectors_first[b_x_idx]}")
    print(f"Using B vector {b_y_idx} for y-direction: {b_vectors_first[b_y_idx]}")
    
    # Validate that B vectors are consistent across k-points (optional check)
    # This is important for irregular grids or boundary effects
    inconsistent_count = 0
    for iq in range(min(10, nq_total)):  # Check first 10 k-points as sample
        b_vec_x_current = b_list[iq, b_x_idx]
        b_vec_y_current = b_list[iq, b_y_idx]
        
        # Check if the direction is still correct
        x_is_x = abs(b_vec_x_current[1]) < tol and abs(b_vec_x_current[2]) < tol and abs(b_vec_x_current[0]) > tol
        y_is_y = abs(b_vec_y_current[0]) < tol and abs(b_vec_y_current[2]) < tol and abs(b_vec_y_current[1]) > tol
        
        if not (x_is_x and y_is_y):
            inconsistent_count += 1
    
    if inconsistent_count > 0:
        print(f"Warning: Found {inconsistent_count} k-points with inconsistent B vector directions.")
        print("This might indicate an irregular grid or boundary effects.")
        print("Results should be interpreted carefully.")
    
    # Get 2D grid dimensions
    if hasattr(qgrid, 'nqx') and hasattr(qgrid, 'nqy') and qgrid.nqx is not None and qgrid.nqy is not None:
        nqx, nqy = qgrid.nqx, qgrid.nqy
    else:
        # Try to infer from total number of q points (assuming square grid)
        nq_total = Mssp.shape[2]
        nqx = nqy = int(np.sqrt(nq_total))
        if nqx * nqy != nq_total:
            # Try to factorize for non-square grids
            found = False
            for test_nqx in range(1, int(np.sqrt(nq_total)) + 1):
                if nq_total % test_nqx == 0:
                    nqx = test_nqx
                    nqy = nq_total // test_nqx
                    found = True
                    break
            if not found:
                raise ValueError(f"Cannot infer 2D grid dimensions from {nq_total} q-points. "
                               f"Please set grid dimensions manually using qgrid.set_grid_dimensions(nqx, nqy)")
        
        print(f"Inferred grid dimensions from Mssp shape: nqx={nqx}, nqy={nqy}")
    
    # Initialize flux array
    flux = np.zeros((nstates, nqx, nqy), dtype=np.float64)
    
    # Find the indices for x and y direction shifts in the Q grid
    def get_neighbor_indices(iq, direction, nqx, nqy):
        """Get neighbor indices for x and y directions
        
        Grid storage: qx is slow index, qy is fast index
        iq = ix * nqy + iy where ix is qx index, iy is qy index
        """
        iy = iq % nqy  # qy index (fast index)
        ix = iq // nqy  # qx index (slow index)
        
        if direction == 'x':
            ix_new = (ix + 1) % nqx if periodic_boundary else ix + 1
            if not periodic_boundary and ix_new >= nqx:
                return None
            return ix_new * nqy + iy
        elif direction == 'y':
            iy_new = (iy + 1) % nqy if periodic_boundary else iy + 1
            if not periodic_boundary and iy_new >= nqy:
                return None
            return ix * nqy + iy_new
        else:
            raise ValueError("Direction must be 'x' or 'y'")
    
    # Compute flux for each plaquette
    # ix is qx index (slow), iy is qy index (fast)
    for ix in range(nqx - (0 if periodic_boundary else 1)):
        for iy in range(nqy - (0 if periodic_boundary else 1)):
            iq = ix * nqy + iy
            
            # Get the four corner points of the plaquette
            iq_x = get_neighbor_indices(iq, 'x', nqx, nqy)  # Q + Δx
            iq_y = get_neighbor_indices(iq, 'y', nqx, nqy)  # Q + Δy
            iq_xy = get_neighbor_indices(iq_x, 'y', nqx, nqy) if iq_x is not None else None  # Q + Δx + Δy
            
            if iq_x is None or iq_y is None or iq_xy is None:
                continue
                
            for istate, state_idx in enumerate(exciton_states):
                # Get overlap matrix elements using the identified B vector indices
                # M(Q, Q+Δx) corresponds to Mssp[state, state, Q, b_x_idx]
                # M(Q, Q+Δy) corresponds to Mssp[state, state, Q, b_y_idx]
                
                M_q_qx = Mssp[state_idx, state_idx, iq, b_x_idx]      # M(Q, Q+Δx)
                M_qx_qxy = Mssp[state_idx, state_idx, iq_x, b_y_idx]  # M(Q+Δx, Q+Δx+Δy)
                M_qy_qxy = Mssp[state_idx, state_idx, iq_y, b_x_idx]  # M(Q+Δy, Q+Δx+Δy)
                M_q_qy = Mssp[state_idx, state_idx, iq, b_y_idx]      # M(Q, Q+Δy)
                
                # Compute U matrices (normalized overlaps)
                U_x_q = M_q_qx / np.abs(M_q_qx) if np.abs(M_q_qx) > 1e-12 else 0
                U_y_qx = M_qx_qxy / np.abs(M_qx_qxy) if np.abs(M_qx_qxy) > 1e-12 else 0
                U_x_qy_inv = np.conj(M_qy_qxy / np.abs(M_qy_qxy)) if np.abs(M_qy_qxy) > 1e-12 else 0
                U_y_q_inv = np.conj(M_q_qy / np.abs(M_q_qy)) if np.abs(M_q_qy) > 1e-12 else 0
                
                # Compute the plaquette product
                plaquette_product = U_x_q * U_y_qx * U_x_qy_inv * U_y_q_inv
                
                # Compute flux as the argument of the plaquette product
                flux[istate, ix, iy] = np.angle(plaquette_product)
    
    # Compute Chern numbers
    chern_numbers = np.zeros(nstates)
    for istate in range(nstates):
        chern_numbers[istate] = np.sum(flux[istate]) / (2 * np.pi)
    
    return flux, chern_numbers


def compute_chern_number_2D(h2p, nnkp_qgrid, exciton_states=None, trange=1, b_vectors=None):
    """
    Compute Chern numbers for exciton states using the plaquette method.
    
    This function first computes the overlap matrix Mssp if not already computed,
    then calculates the flux and Chern numbers.
    
    Parameters
    ----------
    h2p : H2P object
        The two-particle Hamiltonian object containing exciton information
    nnkp_qgrid : object
        Q-point grid object for the calculation
    exciton_states : list or None
        List of exciton state indices to compute Chern numbers for
    trange : int
        Number of exciton states to consider
    b_vectors : ndarray or None
        B vectors from Wannier90. If None, will try to get from nnkp_qgrid.
        
    Returns
    -------
    chern_numbers : ndarray
        Chern numbers for each exciton state
    flux : ndarray
        Flux array for visualization and analysis
    """
    
    # Compute Mssp if not already computed
    if not hasattr(h2p, 'Mssp') or h2p.Mssp is None:
        print("Computing Mssp overlap matrix...")
        Mssp = compute_Mssp(h2p, h2p.kmpgrid, nnkp_qgrid, trange=trange)
    else:
        Mssp = h2p.Mssp
        print("Using existing Mssp overlap matrix...")
    
    # Compute flux and Chern numbers
    print("Computing flux and Chern numbers...")
    flux, chern_numbers = compute_flux_2D(Mssp, nnkp_qgrid, b_vectors=b_vectors, exciton_states=exciton_states)
    
    # Store results in h2p object
    h2p.flux = flux
    h2p.chern_numbers = chern_numbers
    
    print(f"Computed Chern numbers: {chern_numbers}")
    
    return chern_numbers, flux


def plot_flux_and_chern(flux, chern_numbers, exciton_states=None, save_path=None):
    """
    Plot the flux distribution and display Chern numbers.
    
    Parameters
    ----------
    flux : ndarray
        Flux array with shape (nstates, nqx, nqy)
    chern_numbers : ndarray
        Chern numbers for each exciton state
    exciton_states : list or None
        List of exciton state indices. If None, uses all states.
    save_path : str or None
        Path to save the plot. If None, displays the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    except ImportError:
        print("Matplotlib not available. Cannot plot flux.")
        return
    
    nstates = flux.shape[0]
    if exciton_states is None:
        exciton_states = list(range(nstates))
    
    # Create subplots
    ncols = min(3, len(exciton_states))
    nrows = (len(exciton_states) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, state_idx in enumerate(exciton_states):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot flux
        im = ax.imshow(flux[state_idx].T, origin='lower', cmap='RdBu_r', 
                      vmin=-np.pi, vmax=np.pi)
        ax.set_title(f'Exciton State {state_idx}\nChern Number: {chern_numbers[state_idx]:.3f}')
        ax.set_xlabel('Q_x index')
        ax.set_ylabel('Q_y index')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Flux (rad)')
    
    # Hide unused subplots
    for i in range(len(exciton_states), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flux plot saved to {save_path}")
    else:
        plt.show()


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


def find_best_bvector_index(b_vectors_k, direction, tol=1e-6):
    """
    Find the best B-vector index for a given direction at a specific k-point.
    
    Parameters
    ----------
    b_vectors_k : ndarray
        B-vectors for a specific k-point, shape (nb, 3)
    direction : str
        'x' or 'y' direction
    tol : float
        Tolerance for determining if a component is zero
        
    Returns
    -------
    int or None
        Index of the best B-vector, or None if not found
    """
    best_idx = None
    min_magnitude = float('inf')
    
    for ib, b_vec in enumerate(b_vectors_k):
        if direction == 'x':
            # Look for smallest positive x-direction vector (y and z should be ~0)
            if abs(b_vec[1]) < tol and abs(b_vec[2]) < tol and b_vec[0] > tol:
                if abs(b_vec[0]) < min_magnitude:
                    min_magnitude = abs(b_vec[0])
                    best_idx = ib
        elif direction == 'y':
            # Look for smallest positive y-direction vector (x and z should be ~0)
            if abs(b_vec[0]) < tol and abs(b_vec[2]) < tol and b_vec[1] > tol:
                if abs(b_vec[1]) < min_magnitude:
                    min_magnitude = abs(b_vec[1])
                    best_idx = ib
    
    return best_idx


def compute_flux_2D_fixed(Mssp, qgrid, nnkp, exciton_states=None, periodic_boundary=True, 
                          energy_array=None, degeneracy_tol=1e-6, orientation='xy'):
    """
    Simplified and robust flux computation using wannier90 B-vectors.
    
    Always uses:
    - Non-Abelian approach (handles both 1x1 and NxN matrices consistently)
    - Varying B-vectors from wannier90 (nnkp.b_grid)
    - Proper degeneracy handling
    
    Parameters
    ----------
    Mssp : ndarray
        Overlap matrix with shape (nstates, nstates, nq, nb)
    qgrid : object
        Q-point grid object 
    nnkp : object
        NNKP object containing b_grid with varying B-vectors for each k-point
    exciton_states : list or None
        List of exciton state indices to compute flux for
    periodic_boundary : bool
        Whether to use periodic boundary conditions
    energy_array : ndarray, optional
        Energy array E_S(Q) with shape (nstates, nq). If provided, uses
        actual degeneracies. If None, treats each state as separate subspace.
    degeneracy_tol : float
        Tolerance for determining energy degeneracies (default: 1e-6)
        
    Returns
    -------
    flux : ndarray
        Flux array with shape (n_subspaces, nqx, nqy)
    chern_numbers : ndarray
        Chern numbers for each subspace
    """
    
    if exciton_states is None:
        nstates = Mssp.shape[0]
        exciton_states = list(range(nstates))
    else:
        nstates = len(exciton_states)
    
    # Always use non-Abelian approach for consistency
    print("Using non-Abelian approach with wannier90 B-vectors...")
    
    # Validate inputs
    if nnkp is None or not hasattr(nnkp, 'b_grid'):
        raise ValueError("nnkp object with b_grid is required")
    
    nq_total, nb_total = Mssp.shape[2], Mssp.shape[3]
    
    # Find degenerate subspaces
    if energy_array is not None:
        # Validate energy array dimensions
        if energy_array.shape != (Mssp.shape[0], Mssp.shape[2]):
            raise ValueError(f"energy_array shape {energy_array.shape} doesn't match "
                           f"expected ({Mssp.shape[0]}, {Mssp.shape[2]})")
        
        # Find degenerate subspaces for each Q-point
        degenerate_subspaces = find_degenerate_subspaces(energy_array, exciton_states, degeneracy_tol)
        print(f"Found {len(degenerate_subspaces)} degenerate subspaces")
    else:
        # Treat each state as its own subspace (1x1 matrices)
        degenerate_subspaces = [{'states': [i]} for i in exciton_states]
        print(f"Treating {len(degenerate_subspaces)} states as individual subspaces")
    
    # Use wannier90 B-vectors (varying per k-point)
    nk = nq_total  # Assuming nk = nq for the grid
    b_vectors_all = nnkp.b_grid.reshape(nk, nb_total, 3)
    
    print(f"Using varying B-vectors from wannier90: reshaped from {nnkp.b_grid.shape} to {b_vectors_all.shape}")
    
    # Set tolerance for B-vector identification
    tol = 1e-6
    
    # Get 2D grid dimensions
    if hasattr(qgrid, 'nqx') and hasattr(qgrid, 'nqy') and qgrid.nqx is not None and qgrid.nqy is not None:
        nqx, nqy = qgrid.nqx, qgrid.nqy
    else:
        nq_total = Mssp.shape[2]
        nqx = nqy = int(np.sqrt(nq_total))
        if nqx * nqy != nq_total:
            # Try to factorize
            found = False
            for test_nqx in range(1, int(np.sqrt(nq_total)) + 1):
                if nq_total % test_nqx == 0:
                    nqx = test_nqx
                    nqy = nq_total // test_nqx
                    found = True
                    break
            if not found:
                raise ValueError(f"Cannot infer 2D grid dimensions from {nq_total} q-points")
        print(f"Inferred grid dimensions: {nqx} × {nqy}")
    
    # Initialize flux array for degenerate subspaces
    n_subspaces = len(degenerate_subspaces)
    flux = np.zeros((n_subspaces, nqx, nqy), dtype=np.float64)
    
    # Helper function for neighbor indices
    def get_neighbor_indices(iq, direction, nqx, nqy):
        iy = iq % nqy
        ix = iq // nqy
        
        if direction == 'x':
            ix_new = (ix + 1) % nqx if periodic_boundary else ix + 1
            if not periodic_boundary and ix_new >= nqx:
                return None
            return ix_new * nqy + iy
        elif direction == 'y':
            iy_new = (iy + 1) % nqy if periodic_boundary else iy + 1
            if not periodic_boundary and iy_new >= nqy:
                return None
            return ix * nqy + iy_new
        else:
            raise ValueError("Direction must be 'x' or 'y'")
    
    # IMPROVED flux computation with better numerical stability
    valid_plaquettes = 0
    problematic_plaquettes = 0
    
    for ix in range(nqx - (0 if periodic_boundary else 1)):
        for iy in range(nqy - (0 if periodic_boundary else 1)):
            iq = ix * nqy + iy
            
            # Get plaquette corners
            iq_x = get_neighbor_indices(iq, 'x', nqx, nqy)
            iq_y = get_neighbor_indices(iq, 'y', nqx, nqy)
            iq_xy = get_neighbor_indices(iq_x, 'y', nqx, nqy) if iq_x is not None else None
            
            if iq_x is None or iq_y is None or iq_xy is None:
                continue
            
            valid_plaquettes += 1
            
            # Compute flux for this plaquette using non-Abelian approach
            flux_values = compute_nonabelian_flux_plaquette_wannier(
                Mssp, degenerate_subspaces, iq, iq_x, iq_y, iq_xy, 
                b_vectors_all, tol, ix, iy, orientation=orientation
            )
            
            for i_subspace, flux_val in enumerate(flux_values):
                if flux_val is not None:
                    flux[i_subspace, ix, iy] = flux_val
                else:
                    problematic_plaquettes += 1
    
    print(f"Plaquette analysis:")
    print(f"  Valid plaquettes: {valid_plaquettes}")
    print(f"  Problematic plaquettes: {problematic_plaquettes}")
    print(f"  Success rate: {100*(valid_plaquettes-problematic_plaquettes)/valid_plaquettes:.1f}%")
    
    # Compute Chern numbers
    n_subspaces = len(degenerate_subspaces)
    chern_numbers = np.zeros(n_subspaces)
    
    for i_subspace in range(n_subspaces):
        total_flux = np.sum(flux[i_subspace])
        chern_numbers[i_subspace] = total_flux / (2 * np.pi)
        
        subspace_info = degenerate_subspaces[i_subspace]
        print(f"Subspace {i_subspace} (states {subspace_info['states']}):")
        print(f"  Total flux: {total_flux:.6f}")
        print(f"  Chern number: {chern_numbers[i_subspace]:.6f}")
        
        # Validate result
        if abs(chern_numbers[i_subspace]) > 5:
            print(f"  ❌ ERROR: Chern number too large! Check your setup.")
        elif abs(chern_numbers[i_subspace]) > 2:
            print(f"  ⚠️  WARNING: Large Chern number - verify this is correct.")
        elif abs(chern_numbers[i_subspace]) > 0.1:
            print(f"  ✓ Non-trivial Chern number detected.")
        else:
            print(f"  ✓ Small/trivial Chern number.")
    
    return flux, chern_numbers


import numpy as np

def find_degenerate_subspaces(energy_array, exciton_states, degeneracy_tol,
                              relative=False, overlap_threshold=0.5,
                              min_occurrences=1, verbose=False):
    """
    Robust energy-based degenerate-subspace finder.

    Parameters
    ----------
    energy_array : ndarray, shape (nstates, nq)
    exciton_states : list or 1D-array
        Indices of states to consider (integers indexing first axis of energy_array).
    degeneracy_tol : float
        If `relative` is False => absolute tolerance (energy units).
        If `relative` is True  => relative tolerance multiplied by energy scale.
    relative : bool
        Use relative tolerance vs absolute.
    overlap_threshold : float in (0,1]
        When merging groups across Q, minimum fraction of group-members that must
        overlap to be considered the same global subspace.
    min_occurrences : int
        Discard subspaces that appear at fewer than this many Q-points.
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    final_subspaces : list of dicts
        Each dict: {'states': sorted list of state indices (union across Q),
                    'q_points': sorted list of Q indices where a matching group was found,
                    'representative_energy': mean energy across occurrences}
    """
    nstates, nq = energy_array.shape
    exciton_states = list(exciton_states)

    # --- step 1: group per Q using sorted-adjacent-gap clustering ---
    q_degeneracies = [[] for _ in range(nq)]
    for iq in range(nq):
        energies = np.array([energy_array[s, iq] for s in exciton_states])
        order = np.argsort(energies)
        sorted_states = [exciton_states[i] for i in order]
        sorted_energies = energies[order]

        if len(sorted_states) == 0:
            continue

        cur_group = [sorted_states[0]]
        for k in range(1, len(sorted_states)):
            prev_e = sorted_energies[k - 1]
            e_k = sorted_energies[k]
            # choose tolerance (relative scales with typical energy magnitude)
            tol = degeneracy_tol * max(1.0, abs(prev_e)) if relative else degeneracy_tol
            gap = abs(e_k - prev_e)
            if gap <= tol:
                cur_group.append(sorted_states[k])
            else:
                rep_e = float(np.mean([energy_array[s, iq] for s in cur_group]))
                q_degeneracies[iq].append({'states': sorted(cur_group),
                                           'energy': rep_e,
                                           'q_point': iq})
                cur_group = [sorted_states[k]]

        # flush last group
        rep_e = float(np.mean([energy_array[s, iq] for s in cur_group]))
        q_degeneracies[iq].append({'states': sorted(cur_group),
                                   'energy': rep_e,
                                   'q_point': iq})

    # --- step 2: merge groups across Q into global subspaces ---
    global_subspaces = []
    for iq in range(nq):
        for group in q_degeneracies[iq]:
            gset = set(group['states'])
            matched = False
            # try to find an existing global subspace that sufficiently overlaps
            for glob in global_subspaces:
                inter = len(gset & set(glob['states']))
                frac = inter / max(1, len(group['states']))  # fraction of this group's members found
                if frac >= overlap_threshold:
                    # assign group to this global subspace
                    if iq not in glob['q_points']:
                        glob['q_points'].append(iq)
                    # update union of member indices and representative energy
                    glob['states'] = sorted(set(glob['states']) | gset)
                    # running average for representative_energy
                    glob['rep_energies'].append(group['energy'])
                    glob['representative_energy'] = float(np.mean(glob['rep_energies']))
                    matched = True
                    break
            if not matched:
                global_subspaces.append({
                    'states': sorted(group['states']),
                    'q_points': [iq],
                    'rep_energies': [group['energy']],
                    'representative_energy': float(group['energy'])
                })

    # finalize and filter by min_occurrences
    final_subspaces = []
    for g in global_subspaces:
        if len(set(g['q_points'])) >= min_occurrences:
            final_subspaces.append({
                'states': sorted(g['states']),
                'q_points': sorted(set(g['q_points'])),
                'representative_energy': g['representative_energy']
            })

    if verbose:
        print("Degeneracy analysis:")
        for i, s in enumerate(final_subspaces):
            print(f"  Subspace {i}: states {s['states']}, "
                  f"degenerate at {len(s['q_points'])}/{nq} Q-points, "
                  f"E_rep={s['representative_energy']:.6f}")

    return final_subspaces

def compute_nonabelian_flux_plaquette(Mssp, degenerate_subspaces, iq, iq_x, iq_y, iq_xy, 
                                     b_x_idx, b_y_idx, ix, iy):
    """
    Compute flux for a single plaquette in the (possibly non-Abelian) case.

    For each degenerate subspace, compute the Wilson loop around the plaquette.
    If the subspace has dimension > 1, use the determinant of the Wilson loop.
    If dimension = 1, this reduces to the Abelian case.

    Parameters
    ----------
    Mssp : ndarray
        Overlap matrix with shape (nstates, nstates, nq, nb)
    degenerate_subspaces : list
        List of dicts with 'states' defining each degenerate subspace
    iq, iq_x, iq_y, iq_xy : int
        Q-point indices for the plaquette corners
    b_x_idx, b_y_idx : int
        B-vector indices for x and y directions
    ix, iy : int
        Plaquette coordinates (for error reporting)
        
    Returns
    -------
    flux_values : list
        List of flux values for each degenerate subspace (None if problematic)
    """

    def unitarize(M):
        """
        Project a matrix onto the closest unitary via SVD.
        Handles scalar (1x1) case separately.
        """
        if M.shape == (1, 1):  # scalar case
            val = M[0, 0]
            if np.abs(val) == 0:
                return None
            return np.array([[val / np.abs(val)]])  # normalize to unit modulus
        # multi-dimensional case
        U, _, Vh = np.linalg.svd(M, full_matrices=False)
        return U @ Vh

    flux_values = []

    for i_subspace, subspace in enumerate(degenerate_subspaces):
        states = subspace['states']
        n_deg = len(states)

        try:
            # Build overlap matrices for this subspace
            sub_states = np.ix_(states, states)
            U_x_q  = Mssp[sub_states + (iq,  b_x_idx)]
            U_y_qx = Mssp[sub_states + (iq_x, b_y_idx)]
            U_x_qy = Mssp[sub_states + (iq_y, b_x_idx)]
            U_y_q  = Mssp[sub_states + (iq,  b_y_idx)]

            # Project each onto the unitary group
            U_x_q  = unitarize(U_x_q)
            U_y_qx = unitarize(U_y_qx)
            U_x_qy = unitarize(U_x_qy)
            U_y_q  = unitarize(U_y_q)

            if any(M is None for M in [U_x_q, U_y_qx, U_x_qy, U_y_q]):
                flux_values.append(None)
                continue

            # Compute Wilson loop
            Wilson_loop = U_x_q @ U_y_qx @ U_x_qy.conj().T @ U_y_q.conj().T

            # Flux from determinant
            det_W = np.linalg.det(Wilson_loop)
            if np.abs(det_W) < 1e-12:
                flux_values.append(0.0)
                continue

            flux_val = np.angle(det_W)
            flux_values.append(flux_val)

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"WARNING: Linear algebra error in subspace {i_subspace} at plaquette ({ix},{iy}): {e}")
            flux_values.append(None)
        except Exception as e:
            print(f"WARNING: Unexpected error in subspace {i_subspace} at plaquette ({ix},{iy}): {e}")
            flux_values.append(None)

    return flux_values


def apply_parallel_gauge(U_x_q, U_y_qx, U_x_qy, U_y_q, max_iterations=5, tolerance=1e-8):
    """
    Apply parallel gauge to improve numerical stability while preserving Wilson loop.
    
    This procedure applies gauge transformations to make overlap matrices closer 
    to unitary while preserving the Wilson loop determinant (gauge invariant).
    
    Parameters
    ----------
    U_x_q, U_y_qx, U_x_qy, U_y_q : ndarray
        Overlap matrices around the plaquette
    max_iterations : int
        Maximum number of gauge iterations
    tolerance : float
        Convergence tolerance for gauge procedure
        
    Returns
    -------
    U_x_q, U_y_qx, U_x_qy, U_y_q : ndarray
        Gauge-transformed overlap matrices (unitarized)
    """
    def unitarize_svd(M, svd_tol=1e-12):
        """Unitarize matrix using SVD."""
        if M.shape == (1, 1):
            val = M[0, 0]
            if np.abs(val) < svd_tol:
                return None
            return np.array([[val / np.abs(val)]])
        
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        if s[-1] < svd_tol * s[0]:
            return None  # Nearly rank-deficient
        return U @ Vh
    
    # For now, let's use a simpler approach: just unitarize each matrix
    # This preserves the essential physics while improving numerical stability
    
    # The key insight is that we want to project each overlap matrix onto 
    # the unitary group, which removes small eigenvalues that cause numerical issues
    
    U_x_q_unit = unitarize_svd(U_x_q)
    U_y_qx_unit = unitarize_svd(U_y_qx)
    U_x_qy_unit = unitarize_svd(U_x_qy)
    U_y_q_unit = unitarize_svd(U_y_q)
    
    # Check if any unitarization failed
    if any(M is None for M in [U_x_q_unit, U_y_qx_unit, U_x_qy_unit, U_y_q_unit]):
        return None, None, None, None
    
    return U_x_q_unit, U_y_qx_unit, U_x_qy_unit, U_y_q_unit


def compute_nonabelian_flux_plaquette_wannier(Mssp, degenerate_subspaces, iq, iq_x, iq_y, iq_xy, 
                                             b_vectors_all, tol, ix, iy, orientation='xy'):
    """
    Compute flux for a single plaquette in the non-Abelian case using varying B-vectors from wannier90.
    
    This version handles the case where B-vectors vary per k-point, as is the case in real wannier90 calculations.

    Parameters
    ----------
    Mssp : ndarray
        Overlap matrix with shape (nstates, nstates, nq, nb)
    degenerate_subspaces : list
        List of dicts with 'states' defining each degenerate subspace
    iq, iq_x, iq_y, iq_xy : int
        Q-point indices for the plaquette corners
    b_vectors_all : ndarray
        All B-vectors for all k-points, shape (nk, nb, 3)
    tol : float
        Tolerance for B-vector identification
    ix, iy : int
        Plaquette coordinates (for error reporting)
        
    Returns
    -------
    flux_values : list
        List of flux values for each degenerate subspace (None if problematic)
    """
    def unitarize(M, svd_tol=1e-12):
        if M.shape == (1,1):
            val = M[0,0]
            if np.abs(val) < svd_tol:
                return None
            return np.array([[val / np.abs(val)]])
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        if s[-1] < svd_tol * s[0]:
            # nearly rank-deficient -> warn and skip
            return None
        return U @ Vh
    # Find appropriate B-vector indices for each k-point in the plaquette
    b_x_idx_q = find_best_bvector_index(b_vectors_all[iq], 'x', tol)
    b_y_idx_q = find_best_bvector_index(b_vectors_all[iq], 'y', tol)
    b_x_idx_qx = find_best_bvector_index(b_vectors_all[iq_x], 'x', tol)
    b_y_idx_qx = find_best_bvector_index(b_vectors_all[iq_x], 'y', tol)
    b_x_idx_qy = find_best_bvector_index(b_vectors_all[iq_y], 'x', tol)
    b_y_idx_qy = find_best_bvector_index(b_vectors_all[iq_y], 'y', tol)

    # Check if we found valid B-vector indices
    if any(idx is None for idx in [b_x_idx_q, b_y_idx_q, b_x_idx_qx, b_y_idx_qx, b_x_idx_qy, b_y_idx_qy]):
        print(f"WARNING: Could not find appropriate B-vectors for non-Abelian plaquette ({ix},{iy})")
        return [None] * len(degenerate_subspaces)

    flux_values = []

    for i_subspace, subspace in enumerate(degenerate_subspaces):
        states = subspace['states']
        n_deg = len(states)

        # Build overlap matrices for this subspace using varying B-vector indices
        sub_states = np.ix_(states, states)
        U_x_q  = Mssp[sub_states + (iq,  b_x_idx_q)]
        U_y_qx = Mssp[sub_states + (iq_x, b_y_idx_qx)]
        U_x_qy = Mssp[sub_states + (iq_y, b_x_idx_qy)]
        U_y_q  = Mssp[sub_states + (iq,  b_y_idx_q)]

        # Apply parallel gauge to improve numerical stability
        # This makes overlap matrices closer to identity, reducing tiny determinants
        U_x_q, U_y_qx, U_x_qy, U_y_q = apply_parallel_gauge(U_x_q, U_y_qx, U_x_qy, U_y_q)

        if any(M is None for M in [U_x_q, U_y_qx, U_x_qy, U_y_q]):
            flux_values.append(None)
            continue

        # Compute Wilson loop with configurable orientation
        if orientation == 'xy':
            # Counter-clockwise: +x @ +y @ (-x) @ (-y)
            Wilson_loop = U_x_q @ U_y_qx @ U_x_qy.conj().T @ U_y_q.conj().T
        elif orientation == 'yx':
            # Alternate orientation: +y @ +x @ (-y) @ (-x)
            # Uses U_y at q, U_x at qy, reverse-y at qx, reverse-x at q
            Wilson_loop = U_y_q @ U_x_qy @ U_y_qx.conj().T @ U_x_q.conj().T
        else:
            raise ValueError("orientation must be 'xy' or 'yx'")

        # Compute determinant with improved numerical stability
        sign, logabs = np.linalg.slogdet(Wilson_loop)
        
        # More lenient threshold after parallel gauge (should be much more stable now)
        if logabs < -10:  # Improved from -6 due to parallel gauge
            print(f"WARNING: tiny determinant (logabs={logabs:.2e}) at plaquette ({ix},{iy}) after parallel gauge")
            flux_values.append(None)
            continue
            
        # Check conditioning (should be much better after parallel gauge)
        s = np.linalg.svd(Wilson_loop, compute_uv=False)
        condition_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
        if condition_number > 1e10:  # Improved from 1e12 due to parallel gauge
            print(f"WARNING: ill-conditioned Wilson loop (cond ~ {condition_number:.2e}) at ({ix},{iy}) after parallel gauge")
            flux_values.append(None)
            continue

        # Compute flux using the determinant
        if n_deg == 1:
            # For 1x1 case, use the phase directly
            flux_val = np.angle(Wilson_loop[0, 0])
        else:
            # For NxN case, use the phase of the determinant
            flux_val = np.angle(sign) + logabs * 0  # logabs is real, so this is just np.angle(sign)
            
        flux_values.append(flux_val)

    return flux_values


def generate_2D_bvectors(nqx, nqy):
    """
    Generate proper B-vectors for 2D rectangular grids.
    
    For a 2D grid, the minimal set of B-vectors should connect each point
    to its nearest neighbors in a consistent way for Chern number calculation.
    
    Parameters
    ----------
    nqx, nqy : int
        Grid dimensions in x and y directions
        
    Returns
    -------
    bvectors : ndarray
        B-vectors in reciprocal lattice coordinates, shape (nb, 3)
    """
    # For 2D rectangular grids, we need B-vectors that connect to nearest neighbors
    # Standard choice: right, up, and their combinations for proper flux calculation
    
    # Basic nearest neighbor vectors in fractional coordinates
    bvectors = []
    
    # Primary directions (essential for 2D Chern calculation)
    bvectors.append([1.0/nqx, 0.0, 0.0])      # +x direction
    bvectors.append([0.0, 1.0/nqy, 0.0])      # +y direction
    bvectors.append([-1.0/nqx, 0.0, 0.0])     # -x direction  
    bvectors.append([0.0, -1.0/nqy, 0.0])     # -y direction
    
    # Diagonal directions (for completeness and numerical stability)
    bvectors.append([1.0/nqx, 1.0/nqy, 0.0])   # +x+y diagonal
    bvectors.append([1.0/nqx, -1.0/nqy, 0.0])  # +x-y diagonal
    bvectors.append([-1.0/nqx, 1.0/nqy, 0.0])  # -x+y diagonal
    bvectors.append([-1.0/nqx, -1.0/nqy, 0.0]) # -x-y diagonal
    
    return np.array(bvectors)

def unitarize(M):
    """
    Project a matrix onto the closest unitary via SVD.
    Works for scalars (1x1) and matrices.
    """
    if M.shape == (1, 1):  # scalar case
        val = M[0, 0]
        if np.abs(val) == 0:
            return None
        return np.array([[val / np.abs(val)]])  # normalize to unit modulus
    
    # multi-dimensional case
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh