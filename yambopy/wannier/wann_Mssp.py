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

def compute_overlap_kmqmb_kmq(wfdb, nnkp_kgrid, nnkp_qgrid):
    """
    Compute overlap matrix ⟨u_{v'k-Q-B} | u_{vk-Q}⟩ for valence bands.
    This computes overlaps between k-Q-B and k-Q points for all k, Q, B combinations.
    
    Parameters
    ----------
    wfdb : WaveFunction database
        Contains the wavefunctions for BSE bands
    nnkp_kgrid : NNKP_Grids
        K-point grid containing B vectors
    nnkp_qgrid : NNKP_Grids  
        Q-point grid
        
    Returns
    -------
    Mkmqmb_kmq : ndarray
        Overlap matrix with shape (nk, nq, nb, nbands, nbands)
        Mkmqmb_kmq[ik, iq, ib, iv', iv] = ⟨u_{v'k-Q-B} | u_{vk-Q}⟩
    """
    if getattr(wfdb, 'wf_bz', None) is None: 
        wfdb.expand_fullBZ()
    
    # Ensure k-Q-B grid is computed
    if not hasattr(nnkp_kgrid, 'kmqmb_grid_table'):
        print("Computing k-Q-B grid for valence overlaps...")
        nnkp_kgrid.get_kmqmb_grid(nnkp_qgrid, nnkp_kgrid)
    
    nk = nnkp_kgrid.nkpoints
    nq = nnkp_qgrid.nkpoints  
    nb = nnkp_kgrid.nnkpts
    nbands = wfdb.nbands
    
    Mkmqmb_kmq = np.zeros(shape=(nk, nq, nb, nbands, nbands), dtype=np.complex128)
    
    for ik in range(nk):
        for iq in range(nq):
            # Get k-Q index
            ikmq = nnkp_kgrid.kmq_grid_table[ik, iq, 1]
            
            for ib in range(nb):
                # Get k-Q-B index
                ikmqmb = nnkp_kgrid.kmqmb_grid_table[ik, iq, ib, 1]
                
                # Get wavefunctions
                G0_bra = [0, 0, 0]  # Using wannier90 grid
                G0_ket = [0, 0, 0]  # Using wannier90 grid
                
                wfc_kmqmb, gvec_kmqmb = wfdb.get_BZ_wf(ikmqmb)  # k-Q-B
                wfc_kmq, gvec_kmq = wfdb.get_BZ_wf(ikmq)        # k-Q
                
                # Compute overlap ⟨u_{k-Q-B} | u_{k-Q}⟩
                Mkmqmb_kmq[ik, iq, ib] = Mmn_kkp(G0_bra, wfc_kmqmb, gvec_kmqmb, 
                                                  G0_ket, wfc_kmq, gvec_kmq)
    
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


def compute_Mssp(h2p,nnkp_kgrid,nnkp_qgrid,trange=1):
    """
    Compute M_{SS'}(Q,B) = ∑_{cvk} A^{SQ*}_{cvk} A^{S'Q+B}_{c'v'k+B} ⟨u_{ck} | u_{c'k+B}⟩ ⟨u_{v'k-Q-B} | u_{vk-Q}⟩
    Make sure the wfdb used contains only the bse bands.
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
        h2p.Mkmqmb_kmq = compute_overlap_kmqmb_kmq(h2p.wfdb, h2p.kmpgrid, nnkp_qgrid)
    
    for t in range(0,trange):
        for tp in range(0,trange):
            for iq, q in enumerate(nnkp_qgrid.red_kpoints):
                for ib in range(nb):
                    Mssp_ttp = 0
                    iqpb = nnkp_qgrid.qpb_grid_table[iq, ib, 1]  # Q+B index
                    bset = h2p.bse_nc*h2p.bse_nv
                    k = h2p.BSE_table[:, 0]
                    v = h2p.BSE_table[:, 1]
                    c = h2p.BSE_table[:, 2]
                    for ik, iv, ic in zip(k,v,c):  # ∑_{cvk}
                        ikpb = h2p.kmpgrid.kpb_grid_table[ik, ib, 1]  # k+B index
                        for ivp, icp in zip(v[:bset], c[:bset]):

                            # term1: A^{SQ*}_{cvk}
                            term1 = np.conjugate(h2p.h2peigvec_vck[iq, t, h2p.bse_nv - h2p.nv + iv, ic - h2p.nv, ik])
                            # term2: A^{S'Q+B}_{c'v'k+B}
                            term2 = h2p.h2peigvec_vck[iqpb, tp, h2p.bse_nv - h2p.nv + ivp, icp - h2p.nv, ikpb]
                            # term3: ⟨u_{ck} | u_{c'k+B}⟩ = δ_{cc'} = 1 (orthogonality of Bloch states)
                            term3 = 1
                            # term4: ⟨u_{v'k-Q-B} | u_{vk-Q}⟩ using proper valence overlap matrix
                            term4 = h2p.Mkmqmb_kmq[ik, iq, ib, h2p.bse_nv - h2p.nv + ivp, h2p.bse_nv - h2p.nv + iv]
                            
                            Mssp_ttp += np.sum(term1 * term2 * term3 * term4)  # scalar

                    Mssp[t,tp,iq,ib] = Mssp_ttp
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


def example_usage():
    """
    Example of how to use the Chern number calculation functions.
    
    This is a template that shows the typical workflow.
    """
    print("Example usage of Chern number calculation:")
    print("""
    # 1. Set up your H2P object and grids
    h2p = H2P(...)  # Your H2P object
    nnkp_qgrid = ...  # Your Q-point grid
    
    # 2. Compute Chern numbers
    chern_numbers, flux = compute_chern_number_2D(
        h2p, 
        nnkp_qgrid, 
        exciton_states=[0, 1],  # Compute for first two exciton states
        trange=2  # Consider 2 exciton states
    )
    
    # 3. Plot results
    plot_flux_and_chern(flux, chern_numbers, exciton_states=[0, 1])
    
    # 4. Access results
    print(f"Chern numbers: {chern_numbers}")
    print(f"Total Chern number: {np.sum(chern_numbers)}")
    
    # The flux and Chern numbers are also stored in the h2p object:
    # h2p.flux
    # h2p.chern_numbers
    """)


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