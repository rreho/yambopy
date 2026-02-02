##
## Authors: MN (FP adapted, SB optimized)
##
import numpy as np
import os
from netCDF4 import Dataset
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.dbs.dipolesdb import YamboDipolesDB
from yambopy.bse.exciton_matrix_elements import exciton_X_matelem
from yambopy.bse.rotate_excitonwf import rotate_exc_wf
from tqdm import tqdm
import torch


# ---------- your helpers ----------
def make_kpositive(klist, tol=1e-6):
    kpos = klist - np.floor(klist)
    return (kpos + tol) % 1

def find_kindx(tree, kpt_search, tol=1e-5):
    kpt_search = make_kpositive(kpt_search)
    dist, idx = tree.query(kpt_search, workers=1)
    if np.any(dist > tol):
        raise RuntimeError("Kpoint not found within tolerance")
    return idx

def scatter_reduced_to_full_torch(vec_red: torch.Tensor,
                                  keep_idx_t: torch.Tensor,
                                  full_shape):
    """
    vec_red: (nS, ntrans_red) complex
    returns: (nS, nk, nc, nv)
    """
    nS = vec_red.shape[0]
    nk, nc, nv = map(int, full_shape)
    out = torch.zeros((nS, nk * nc * nv), dtype=vec_red.dtype, device=vec_red.device)
    out[:, keep_idx_t] = vec_red
    return out.view(nS, nk, nc, nv)

# ---------- FULL (non-reduced) GPU kernel, streaming to out slice ----------
@torch.no_grad()
def ex_ph_mat_full_gpu_write_slice(
    wfc_k_q: np.ndarray,       # (nS, nk, nc, nv) at Q+q
    wfc_k:   np.ndarray,       # (nS, nk, nc, nv) at Q
    elph_mat: np.ndarray,      # (nmodes, nk, nb, nb) for THIS phonon q
    qpt_exe: np.ndarray,
    qpt_ph:  np.ndarray,
    kpts: np.ndarray,
    ktree,
    out_mm: np.ndarray,        # memmap slice parent
    out_index: tuple,          # (iQpos, iqpos)
    device="cuda",
    cdtype=np.complex64,
    mode_chunk=8,
    S_block=1024,
    use_pinned=True,
):
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    if cdtype == np.complex64:
        cplx = torch.complex64
    elif cdtype == np.complex128:
        cplx = torch.complex128
    else:
        raise ValueError("cdtype must be np.complex64 or np.complex128")

    nmodes = int(elph_mat.shape[0])
    nS, nk, nc, nv = map(int, wfc_k.shape)

    # k mappings (CPU)
    idx_k_minus_Q_minus_q = find_kindx(ktree, kpts - qpt_ph[None, :] - qpt_exe[None, :])
    idx_k_minus_q         = find_kindx(ktree, kpts - qpt_ph[None, :])

    # to torch once
    A  = torch.as_tensor(wfc_k,   dtype=cplx, device=dev)     # (nS,nk,nc,nv)
    Aq = torch.as_tensor(wfc_k_q, dtype=cplx, device=dev)     # (nS,nk,nc,nv)
    AqT_conj = Aq.reshape(nS, -1).conj().transpose(0, 1).contiguous()  # (nk*nc*nv, nS)

    # out view: (nmodes,nS,nS)
    out_view = out_mm[out_index]

    # elph pinned
    if use_pinned and dev.type == "cuda":
        elph_pinned = torch.from_numpy(elph_mat).to(cplx).pin_memory()
        def get_elph_chunk(m0, m1):
            return elph_pinned[m0:m1].to(dev, non_blocking=True)
    else:
        def get_elph_chunk(m0, m1):
            return torch.as_tensor(elph_mat[m0:m1], dtype=cplx, device=dev)

    for m0 in range(0, nmodes, mode_chunk):
        m1 = min(m0 + mode_chunk, nmodes)
        mch = m1 - m0

        elph_ch = get_elph_chunk(m0, m1)  # (mch,nk,nb,nb)
        gcc = elph_ch[:, idx_k_minus_q,         nv:, nv:]   # (mch,nk,nc,nc)
        gvv = elph_ch[:, idx_k_minus_Q_minus_q, :nv, :nv]   # (mch,nk,nv,nv)

        A_e = A[:, idx_k_minus_q, ...].contiguous()         # (nS,nk,nc,nv)

        # tmp_e: (mch,nS,nk,nc,nv)
        tmp_e = torch.einsum("Skiv,mkci->mSkcv", A_e, gcc)
        tmp_h = torch.einsum("Skci,mkiv->mSkcv", A,  gvv)
        tmp_e.sub_(tmp_h)
        del tmp_h, gcc, gvv, elph_ch

        tmp_flat = tmp_e.reshape(mch, nS, -1)               # (mch,nS,nk*nc*nv)
        del tmp_e

        for S0 in range(0, nS, S_block):
            S1 = min(S0 + S_block, nS)
            out_block = torch.matmul(tmp_flat, AqT_conj[:, S0:S1])  # (mch,nS,Sb)
            out_view[m0:m1, :, S0:S1] = out_block.detach().cpu().numpy().astype(cdtype, copy=False)
            del out_block

        del tmp_flat
        if dev.type == "cuda":
            torch.cuda.synchronize()

# ---------- REDUCED GPU kernel, streaming to out slice ----------
@torch.no_grad()
def ex_ph_mat_reduced_gpu_write_slice(
    wfc_k_q_red: np.ndarray,   # (nS, ntrans_red) at Q+q
    wfc_k_red:   np.ndarray,   # (nS, ntrans_red) at Q
    elph_mat:    np.ndarray,   # (nmodes, nk, nb, nb) for THIS phonon q
    qpt_exe, qpt_ph, kpts, ktree,
    keep_idx:    np.ndarray,   # integer indices into flattened (nk*nc*nv)
    full_shape,                # (nk, nc, nv)
    out_mm:      np.ndarray,
    out_index:   tuple,
    device="cuda",
    cdtype=np.complex64,
    mode_chunk=8,
    S_block=1024,
    use_pinned=True,
):
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    if cdtype == np.complex64:
        cplx = torch.complex64
    elif cdtype == np.complex128:
        cplx = torch.complex128
    else:
        raise ValueError("cdtype must be np.complex64 or np.complex128")

    nmodes = int(elph_mat.shape[0])
    nS = int(wfc_k_red.shape[0])
    nk, nc, nv = map(int, full_shape)

    idx_k_minus_Q_minus_q = find_kindx(ktree, kpts - qpt_ph[None, :] - qpt_exe[None, :])
    idx_k_minus_q         = find_kindx(ktree, kpts - qpt_ph[None, :])

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long, device=dev)

    A_red  = torch.as_tensor(wfc_k_red,   dtype=cplx, device=dev)
    Aq_red = torch.as_tensor(wfc_k_q_red, dtype=cplx, device=dev)
    AqT_conj = Aq_red.conj().transpose(0, 1).contiguous()          # (ntrans_red,nS)

    A_full = scatter_reduced_to_full_torch(A_red, keep_idx_t, (nk, nc, nv))  # (nS,nk,nc,nv)

    out_view = out_mm[out_index]  # (nmodes,nS,nS)

    if use_pinned and dev.type == "cuda":
        elph_pinned = torch.from_numpy(elph_mat).to(cplx).pin_memory()
        def get_elph_chunk(m0, m1):
            return elph_pinned[m0:m1].to(dev, non_blocking=True)
    else:
        def get_elph_chunk(m0, m1):
            return torch.as_tensor(elph_mat[m0:m1], dtype=cplx, device=dev)

    for m0 in range(0, nmodes, mode_chunk):
        m1 = min(m0 + mode_chunk, nmodes)
        mch = m1 - m0

        elph_ch = get_elph_chunk(m0, m1)
        gcc = elph_ch[:, idx_k_minus_q,         nv:, nv:]
        gvv = elph_ch[:, idx_k_minus_Q_minus_q, :nv, :nv]

        tmp_e = torch.einsum("Skiv,mkci->mSkcv", A_full, gcc)
        tmp_h = torch.einsum("Skci,mkiv->mSkcv", A_full, gvv)
        tmp_e.sub_(tmp_h)
        del tmp_h, gcc, gvv, elph_ch

        tmp_red = tmp_e.reshape(mch, nS, -1).index_select(dim=2, index=keep_idx_t)
        del tmp_e

        for S0 in range(0, nS, S_block):
            S1 = min(S0 + S_block, nS)
            out_block = torch.matmul(tmp_red, AqT_conj[:, S0:S1])   # (mch,nS,Sb)
            out_view[m0:m1, :, S0:S1] = out_block.detach().cpu().numpy().astype(cdtype, copy=False)
            del out_block

        del tmp_red
        if dev.type == "cuda":
            torch.cuda.synchronize()


# ---------- inference helpers (keeps your public API clean) ----------
def _infer_keep_idx_and_full_shape(lattice, elphdb, wfdb, BSE_dir):
    """
    Tries to infer keep_idx and full_shape without extra user inputs.
    Priority:
      1) wfdb.keep_idx / wfdb.full_shape
      2) elphdb.keep_idx / elphdb.full_shape
      3) BSE_dir contains a .npz with 'keep_mask' or 'keep_idx' and maybe full_shape
    """
    # 1) attributes on wfdb/elphdb
    for obj in (wfdb, elphdb, lattice):
        if hasattr(obj, "keep_idx") and hasattr(obj, "full_shape"):
            return np.asarray(obj.keep_idx, dtype=np.int64), tuple(obj.full_shape)

    # 2) try to find a mask file in BSE_dir
    if BSE_dir is not None and os.path.isdir(BSE_dir):
        cand = []
        for fn in os.listdir(BSE_dir):
            if fn.endswith(".npz") and ("mask" in fn or "rediag" in fn or "keep" in fn):
                cand.append(os.path.join(BSE_dir, fn))
        cand.sort()
        for fn in cand:
            try:
                z = np.load(fn, allow_pickle=True)
                if "keep_idx" in z:
                    keep_idx = np.asarray(z["keep_idx"], dtype=np.int64)
                elif "keep_mask" in z:
                    keep_idx = np.where(np.asarray(z["keep_mask"]).ravel())[0].astype(np.int64)
                else:
                    continue

                if "full_shape" in z:
                    full_shape = tuple(map(int, z["full_shape"]))
                    return keep_idx, full_shape
                # if full_shape not present, we cannot safely infer nc,nv
            except Exception:
                pass

    raise RuntimeError(
        "Reduced exciton vectors detected, but keep_idx/full_shape could not be inferred.\n"
        "Add attributes wfdb.keep_idx + wfdb.full_shape (nk,nc,nv), or store them in a .npz in BSE_dir "
        "with keys 'keep_idx' (or 'keep_mask') and 'full_shape'."
    )


def exciton_phonon_matelem_gpu(
    lattice,
    elphdb,
    wfdb,
    BSE_dir=None,
    neigs=10,
    dmat_mode="save",
    Qrange=None,                 # [start, stop] in wfdb.kBZ indexing
    exph_file="ws2-Ex-ph.npy",
    overwrite=False,
    device="cuda",
    cdtype=np.complex64,
    mode_chunk=8,
    S_block=1024,
):
    """
    Writes/returns Ex-ph matrix elements for all Q in Qrange and all phonon q in elphdb:

      exph[ iQ, iq, nu, S, S' ]  with shape (nQ, nq, nmodes, neigs, neigs)

    Uses:
      - rotate_Akcv_Q(...) (must exist in your environment)
      - elphdb.read_iq(iq, convention='standard') -> (ph_eig, elph_mat)
      - elphdb.qpoints[iq] phonon q-point (reduced coords)
      - wfdb.kBZ and wfdb.ktree
      - Dmats: supply/compute according to your dmat_mode workflow outside or inside rotate_Akcv_Q
    """

    if (not overwrite) and os.path.exists(exph_file):
        # memory-map existing npy for fast access
        return np.load(exph_file, mmap_mode="r")

    if Qrange is None:
        Qrange = [0, wfdb.nkBZ]  # same as your example
    Q0, Q1 = int(Qrange[0]), int(Qrange[1])
    Q_indices = list(range(Q0, Q1))

    nq = int(elphdb.nq)

    # Determine nmodes from first q
    _, elph0 = elphdb.read_iq(0, convention="standard")
    elph0 = elph0.transpose(1, 0, 2, 4, 3)   # your convention -> (nmodes,nk,nb,nb)
    nmodes = int(elph0.shape[0])

    nQ = len(Q_indices)
    nS = int(neigs)

    # create .npy as a memmap with header (no extra conversion step)
    out_mm = np.lib.format.open_memmap(
        exph_file, mode="w+", dtype=cdtype, shape=(nQ, nq, nmodes, nS, nS)
    )

    # Try to infer reduced-space metadata once (only used if needed)
    keep_idx = None
    full_shape = None
    exdbs = []
    for iQ in tqdm(range(Qrange[0],Qrange[1])):
        # wfdb.kpoints_indexes[iQ]
        filename = 'ndb.BS_diago_Q%d' % (lattice.kpoints_indexes[iQ]+1)
        excdb = YamboExcitonDB.from_db_file(lattice,filename=filename,folder=BSE_dir,\
                                            Load_WF=True, neigs=neigs)
        exdbs.append(excdb)

    # get D matrices
    Dmats = save_or_load_dmat(wfdb,mode=dmat_mode,dmat_file='Dmats.npy')
    # Main loops
    for iQ_pos, iQ in enumerate(tqdm(Q_indices, desc="Q")):
        Q_in = np.asarray(wfdb.kBZ[iQ], dtype=float)

        # Build Ak(Q)
        # You already have rotate_Akcv_Q in your codebase; keep its behavior unchanged.
        Ak = rotate_Akcv_Q(wfdb, exdbs, Q_in, Dmats, folder=BSE_dir) 
        Ak = np.asarray(Ak)
        Ak = Ak[:nS]

        # determine full vs reduced once
        is_reduced = (Ak.ndim == 2)   # (nS,ntrans_red)
        is_full    = (Ak.ndim == 4)   # (nS,nk,nc,nv)
        if not (is_reduced or is_full):
            raise ValueError(f"rotate_Akcv_Q returned unexpected shape {Ak.shape}")

        if is_reduced and (keep_idx is None or full_shape is None):
            keep_idx, full_shape = _infer_keep_idx_and_full_shape(lattice, elphdb, wfdb, BSE_dir)

        for iq in tqdm(range(nq), desc="q", leave=False):
            ph_eig, elph_mat = elphdb.read_iq(iq, convention="standard")
            elph_mat = elph_mat.transpose(1, 0, 2, 4, 3)  # -> (nmodes,nk,nb,nb)
            qpt_ph = np.asarray(elphdb.qpoints[iq], dtype=float)

            # Build Ak(Q+q)
            Akq = rotate_Akcv_Q(wfdb, exdbs, Q_in + qpt_ph, Dmats, folder=BSE_dir)  # adapt args
            Akq = np.asarray(Akq)[:nS]

            if is_full:
                ex_ph_mat_full_gpu_write_slice(
                    wfc_k_q=Akq,
                    wfc_k=Ak,
                    elph_mat=elph_mat,
                    qpt_exe=Q_in,
                    qpt_ph=qpt_ph,
                    kpts=np.asarray(wfdb.kBZ, dtype=float),
                    ktree=wfdb.ktree,
                    out_mm=out_mm,
                    out_index=(iQ_pos, iq),
                    device=device,
                    cdtype=cdtype,
                    mode_chunk=mode_chunk,
                    S_block=S_block,
                    use_pinned=True,
                )
            else:
                ex_ph_mat_reduced_gpu_write_slice(
                    wfc_k_q_red=Akq,
                    wfc_k_red=Ak,
                    elph_mat=elph_mat,
                    qpt_exe=Q_in,
                    qpt_ph=qpt_ph,
                    kpts=np.asarray(wfdb.kBZ, dtype=float),
                    ktree=wfdb.ktree,
                    keep_idx=keep_idx,
                    full_shape=full_shape,
                    out_mm=out_mm,
                    out_index=(iQ_pos, iq),
                    device=device,
                    cdtype=cdtype,
                    mode_chunk=mode_chunk,
                    S_block=S_block,
                    use_pinned=True,
                )

    out_mm.flush()
    return np.load(exph_file, mmap_mode="r")


def exciton_phonon_matelem(latdb,elphdb,wfdb,Qrange=None,BSE_dir='bse',BSE_Lin_dir=None,
                           neigs=-1,dmat_mode='run',save_files=True,exph_file='Ex-ph.npy',overwrite=False,
                           save_excitons=False,save_lattice=False,save_dipoles=False):
    """
    This function calculates the exciton-phonon matrix elements

    - Q is the exciton momentum
    - q is the phonon momentum
    - exc_in represent the "initial" exciton states in the scattering process (at mom. Q)
    - exc_out represents the "final" exciton states in the scattering process (at mom. Q+q)

    Parameters
    ----------
    latdb : YamboLatticeDB
        The YamboLatticeDB object which contains the lattice information.
    elphdb : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    BSE_dir : str, optional
        The name of the folder which contains the BSE calculation. Default is 'bse'.
    BSE_Lin_dir : str, optional
        The name of the folder which contains the BSE Q=0 calculation (for optical spectra). Default is BSE_dir.
    Qrange : int list or 'full', optional
        Exciton Qpoint index range [iQ_initial, iQ_final] (python counting). 
        If 'full', it covers all k-points in the FBZ. Default is [0,1] (Gamma point only).
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    dmat_mode : str, optional
        If 'save', print dmats on .npy file for faster recalculation. If 'load', load from .npy file. Else, calculate Dmats at runtime.
    save_files : bool, optional
        If True, the matrix elements will be saved in .npz file `exph_file` with metadata. Default is True.
    overwrite : bool, optional
        If False and `exph_file` is found, the matrix elements will be loaded from file. Default is False.
    save_excitons : bool, optional
        If True, save all BSE eigenvectors and energies for the full BZ in a single `excitons.nc` file. Default is False.
    save_lattice : bool, optional
        If True, save lattice and symmetry information in a single `lattice.nc` file. Default is False.
    save_dipoles : bool, optional
        If True, save expanded dipoles in `dipoles.nc` and exciton dipoles in `excitons.nc`. Default is False.
    """
    if Qrange is None: Qrange = [0,1]
    if Qrange == 'full': Qrange = [0, wfdb.nkBZ]

    # Check if we just need to load
    if exph_file.endswith('.nc'): exph_file_path = exph_file
    else: exph_file_path = exph_file if exph_file.endswith('.npz') else exph_file.replace('.npy', '.npz')

    if os.path.exists(exph_file_path) and overwrite==False:
        print(f'Loading EXCPH matrices from {exph_file_path}...')
        if exph_file_path.endswith('.nc'):
            with Dataset(exph_file_path, 'r') as f:
                G_tmp = f.variables['G'][:]
                exph_mat_loaded = G_tmp[...,0] + 1j*G_tmp[...,1]
                # If it was saved with nQ=1, G might be 4D or 5D depending on how it was saved.
                # Here we return it as is.
        else:
            data = np.load(exph_file_path)
            exph_mat_loaded = data['G']
        return exph_mat_loaded

    # Load exc dbs
    exdbs = []
    excitons_nc_path = os.path.join(BSE_dir, 'excitons.nc')
    if os.path.exists(excitons_nc_path):
        print(f'Loading excitons from {excitons_nc_path}...')
        exdbs = YamboExcitonDB.load_excitons_nc(latdb, excitons_nc_path)
    else:
        for ik in range(wfdb.nkpoints):
            filename = 'ndb.BS_diago_Q%d' % (ik+1)
            excdb = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=BSE_dir,\
                                                Load_WF=True, neigs=neigs)
            exdbs.append(excdb)
            #
            # NM : Add a sanity check to avoid a disasterous consequence
            # if the user gives wrong bse band indices.
            min_bnd_bse = np.min(excdb.unique_vbands)
            max_bnd_bse = np.max(excdb.unique_cbands)+1
            assert (wfdb.min_bnd == min_bnd_bse) and ((wfdb.min_bnd + wfdb.nbands) == max_bnd_bse), \
                print("Error: BSE bands mismatch. Given bands range : [%d, %d]. " %(
                    wfdb.min_bnd,wfdb.min_bnd + wfdb.nbands) +
                    "Bse band range found (expected) : [%d %d]" %( min_bnd_bse,max_bnd_bse))

    # get D matrices
    Dmats = save_or_load_dmat(wfdb,mode=dmat_mode,dmat_file='Dmats.npy')

    if save_dipoles:
        dipoles_path = os.path.join(BSE_dir, 'ndb.dipoles')
        if not os.path.exists(dipoles_path):
             dipoles_path = 'ndb.dipoles'
        if os.path.exists(dipoles_path):
             print(f'Loading dipoles from {dipoles_path}...')
             # Load dipoles, don't project, expand to FBZ
             # Use the bands from the first exciton DB in the list
             bse_bands = exdbs[0].bs_bands
             dipdb = YamboDipolesDB.from_db_file(latdb, filename=dipoles_path, bands_range=bse_bands, project=False, expand=True)
             print(f'Saving expanded dipoles to dipoles.nc (bands: {bse_bands})...')
             nq = Qrange[1] - Qrange[0]
             dipdb.save_nc('dipoles.nc', nq=nq)
        else:
             print('[WARNING] ndb.dipoles not found. Dipoles will not be saved.')
             save_dipoles = False

    # Calculation
    print('Calculating EXCPH matrix elements...')
    exph_mat = []
    Q_points = []
    for iQ in tqdm(range(Qrange[0],Qrange[1])):
        Q_in = wfdb.kBZ[iQ]
        Q_points.append(Q_in)
        exph_mat.append( exciton_phonon_matelem_iQ(elphdb,wfdb,exdbs,Dmats,\
                                                   BSE_Lin_dir=BSE_Lin_dir,Q_in=Q_in,neigs=neigs) )
    # IO
    if len(exph_mat)<2: exph_mat = exph_mat[0] # single Q-point calculation (suppress axis)
    else:               exph_mat = np.array(exph_mat) #[nQ,nq,nmodes,nexc_in (Qexc),nexc_out (Qexc+q)]
    
    if save_excitons:
        # Compute exciton dipoles at Q=0 if possible
        exc_dipoles_0 = None
        if save_dipoles:
             idx_gamma_bz = wfdb.kptBZidx(np.zeros(3))
             if len(exdbs) == wfdb.nkBZ: # FBZ list
                 gamma_db = exdbs[idx_gamma_bz]
             else: # IBZ list
                 iQ_gamma_ibz = latdb.kpoints_indexes[idx_gamma_bz]
                 gamma_db = exdbs[iQ_gamma_ibz]
             
             # Compute exciton dipoles at Gamma: D_S = sum_{kcv} A_{kcv} d_{kcv}
             # ydip for Gamma only
             ydip_gamma = YamboDipolesDB.from_db_file(latdb, filename=dipoles_path, 
                                                      bands_range=bse_bands, 
                                                      project=False, expand=False,debug=True)
             
             # Expand dipoles manually to full BZ (without square matrix reshaping)
             rot_mats = latdb.sym_car[latdb.kmap[:, 1], ...]
             dip_expanded = np.einsum('kij,kjcv->kicv', rot_mats, ydip_gamma.dipoles[latdb.kmap[:, 0]], optimize=True)
             time_rev_s = (latdb.kmap[:, 1] >= len(latdb.sym_car) / (1 + int(latdb.time_rev)))
             dip_expanded[time_rev_s] = dip_expanded[time_rev_s].conj()
             
             # Apply physical normalization (1/Nk)
             dip_expanded /= latdb.nkpoints

             BS_wfc = gamma_db.get_Akcv() # [nexcs, nblks, nspin, nk, nc, nv]
             if BS_wfc.shape[1] == 1 and BS_wfc.shape[2] == 1: # TDA, no spin
                 BS_wfc = np.squeeze(BS_wfc, axis=(1, 2)) # [nexcs, nk, nc, nv]
                 # dip_expanded shape: [nkBZ, 3, nc, nv]
                 # Expand BS_wfc to full BZ if needed
                 if BS_wfc.shape[1] != dip_expanded.shape[0]:
                     BS_wfc = BS_wfc[:, latdb.kmap[:, 0], ...]
                 
                 # Sanity check for dimensions
                 if BS_wfc.shape[2:] != dip_expanded.shape[2:]:
                     print(f'[WARNING] Dimension mismatch: BS_wfc {BS_wfc.shape[2:]} vs Dipoles {dip_expanded.shape[2:]}. Slicing may be required.')
                     # Use the minimum common dimensions
                     nc_min = min(BS_wfc.shape[2], dip_expanded.shape[2])
                     nv_min = min(BS_wfc.shape[3], dip_expanded.shape[3])
                     BS_wfc = BS_wfc[..., :nc_min, :nv_min]
                     dip_expanded = dip_expanded[..., :nc_min, :nv_min]

                 exc_dipoles_0 = np.einsum('nkcv,kicv->in', BS_wfc, dip_expanded, optimize=True)
                 # result shape: [3, nexcs]
             else:
                 print('[WARNING] Exciton dipoles calculation only supported for TDA and no-spin-pol. Skipping.')

        print('Expanding and saving excitons for the full BZ to excitons.nc...')
        full_exdbs = []
        for iQ in tqdm(range(wfdb.nkBZ), desc='Save excitons to full BZ in excitons.nc '):
            Qpt = wfdb.kBZ[iQ] # reduced
            # Get rotated Akcv
            Ak_rot = rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats)
            
            # Identify IBZ index
            idx_BZQ = wfdb.kptBZidx(Qpt)
            iQ_iBZ = latdb.kpoints_indexes[idx_BZQ]
            ibz_db = exdbs[iQ_iBZ]
            
            # Create rotated YamboExcitonDB
            # Flatten Ak_rot back to eigenvectors format
            eigenvectors_rot = ibz_db.flatten_Akcv(Ak_rot)
            
            # Rotate exc_dipoles_0 if present
            exc_dipoles_Q = None
            if exc_dipoles_0 is not None:
                isym = latdb.symmetry_indexes[idx_BZQ]
                rot_mat = latdb.sym_car[isym]
                # Rotate the vector components
                exc_dipoles_Q = np.einsum('ij,jn->in', rot_mat, exc_dipoles_0)
                # Time reversal conjugation
                if isym >= len(latdb.sym_car) / (1 + int(latdb.time_rev)):
                    exc_dipoles_Q = exc_dipoles_Q.conj()
            
            full_exdb = YamboExcitonDB(latdb, str(iQ+1), ibz_db.eigenvalues, 
                                       ibz_db.l_residual, ibz_db.r_residual,
                                       spin_pol=ibz_db.spin_pol, red_qpoint=Qpt,
                                       table=ibz_db.table, eigenvectors=eigenvectors_rot,
                                       exc_dipoles=exc_dipoles_Q,
                                       electronic_dipoles=dip_expanded if save_dipoles else None)
            full_exdbs.append(full_exdb)

        YamboExcitonDB.save_excitons_nc(full_exdbs, 'excitons.nc')
    
    if save_lattice:
        print('Saving lattice to lattice.nc...')
        latdb.save_nc('lattice.nc')

    if save_files:
        if exph_file.endswith('.nc'):
            exph_file_path = exph_file
            print(f'Excph coupling file saved to {exph_file_path}')
            with Dataset(exph_file_path, 'w', format='NETCDF4') as f:
                # Dimensions
                f.createDimension('complex', 2)
                f.createDimension('Q_out', elphdb.nq)
                f.createDimension('q_coords', 3)
                f.createDimension('Q_init', len(Q_points))
                f.createDimension('Q_coords', 3)
                
                # Exph matrix elements
                # exph_mat shape: [nQ, nq, nmodes, nexc_in, nexc_out]
                if exph_mat.ndim == 5: dims_G = ['Q_init', 'Q_out'] 
                else:   dims_G = ['Q_out']

                for i, dim in enumerate(exph_mat.shape[len(dims_G):]):
                    dim_name = f'dim_G_{i}'
                    f.createDimension(dim_name, dim)
                    dims_G.append(dim_name)
                dims_G.append('complex')
                
                G_var = f.createVariable('G', 'f8', dims_G)
                G_var[..., 0] = exph_mat.real
                G_var[..., 1] = exph_mat.imag
                
                # Q_init
                Q_init_var = f.createVariable('Q_init', 'f8', ('Q_init', 'Q_coords'))
                Q_init_var[:] = np.array(Q_points)
                
                # Q_out
                Q_out_var = f.createVariable('Q_out', 'f8', ('Q_out', 'Q_coords'))
                Q_out_var[:] = np.array(Q_points) + elphdb.qpoints
                
        else:
            exph_file_path = exph_file if exph_file.endswith('.npz') else exph_file.replace('.npy', '.npz')
            print(f'Excph coupling file saved to {exph_file_path}')
            np.savez(exph_file_path, G=exph_mat, Q_init=np.array(Q_points), Q_out=np.array(Q_points)+elphdb.qpoints)
    
    return exph_mat

def exciton_phonon_matelem_iQ(elphdb,wfdb,exdbs,Dmats,BSE_Lin_dir=None,
                              Q_in=np.zeros(3),neigs=-1,dmat_mode='run'): 
    """
    This function calculates the exciton-phonon matrix element per Q 

    - Q is the exciton momentum
    - q is the phonon momentum
    - exc_in represent the "initial" exciton states in the scattering process (at mom. Q)
    - exc_out represents the "final" exciton states in the scattering process (at mom. Q+q)

    Parameters
    ----------
    elphdb : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    exdbs : YamboExcitonDB list
        List of Q+q YamboExcitonDB objects containing the BSE calculation
    BSE_Lin_dir : str, optional
        The name of the folder which contains the BSE q=0 calculation (for optical spectra). Default is exdbs[Q].
    Q_in : np.ndarray, optional
        Excitonic momentum in reduced units. Default np.array([0.0,0.0,0.0]) 
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    """
    latdb = wfdb.ydb
    # Determine Lkind(in)
    Ak = rotate_Akcv_Q(wfdb, exdbs, Q_in, Dmats, folder=BSE_Lin_dir)
    # Compute ex-ph
    exph_mat = []
    bse_bnds_range = [wfdb.min_bnd,wfdb.min_bnd + wfdb.nbands]
    for iq in range(elphdb.nq):
        ph_eig, elph_mat = elphdb.read_iq(iq,bands_range=bse_bnds_range,convention='standard')
        elph_mat = elph_mat.transpose(1,0,2,4,3)
        #
        Akq = rotate_Akcv_Q(wfdb, exdbs, Q_in + elphdb.qpoints[iq], Dmats) # q+Q
        tmp_exph = exciton_X_matelem(Q_in, elphdb.qpoints[iq], \
                                     Akq, Ak, elph_mat, wfdb.kBZ, \
                                     contribution='b', diagonal_only=False, ktree=wfdb.ktree)
        exph_mat.append(tmp_exph)

    ## 0.5 for Ry to Ha
    exph_mat = 0.5 * np.array(exph_mat).transpose(0,1,3,2) #[nq,nmodes,nexc_in (Qexc),nexc_out (Qexc+q)]

    return exph_mat

def save_or_load_dmat(wfdb, mode='run', dmat_file='Dmats.npy'):
    """
    Save or load Dmats to/from .npy file `dmat_file` for faster recalculation.

     If mode=='save', print dmats on .npy file for faster recalculation. 
     If mode=='load', load from .npy file. 
     Else, calculate Dmats at runtime.
    """
    if dmat_file[-4:]!='.npy': dmat_file = dmat_file+'.npy'
    if mode=='save':
        print('Saving D matrices...')
        Dmats = wfdb.Dmat()
        np.save(dmat_file,Dmats)
        return Dmats
    elif mode=='load': 
        print('Loading D matrices...')
        if not os.path.exists(dmat_file):
            raise FileNotFoundError(f"Cannot load '{dmat_file}' - file does not exist.")
        Dmats_loaded = np.load(dmat_file)
        return Dmats_loaded
    else:
        return wfdb.Dmat()


def rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats, folder=None):
    '''
    Qpt reduced coordinates in BZ
    '''
    latdb = wfdb.ydb
    idx_BZQ = wfdb.kptBZidx(Qpt)
    
    # If exdbs is already expanded to FBZ, just return the wavefunction
    if len(exdbs) == wfdb.nkBZ and folder is None:
        return exdbs[idx_BZQ].get_Akcv()

    iQ_isymm = latdb.symmetry_indexes[idx_BZQ]
    iQ_iBZ = latdb.kpoints_indexes[idx_BZQ]
    trev  = (iQ_isymm >= len(latdb.sym_car) / (1 + int(np.rint(latdb.time_rev))))
    symm_mat_red = latdb.lat@latdb.sym_car[iQ_isymm]@np.linalg.inv(latdb.lat)
    exe_iQIBZ = wfdb.kpts_iBZ[iQ_iBZ]
    #
    if folder is not None:
        neigs = len(exdbs[0].eigenvalues)
        filename = 'ndb.BS_diago_Q%d' % (iQ_iBZ+1)
        excdbin = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=folder,\
                                              Load_WF=True, neigs=neigs)
        AQibz = excdbin.get_Akcv()
    else : AQibz = exdbs[iQ_iBZ].get_Akcv()
    #
    AQ_rot = rotate_exc_wf(AQibz,symm_mat_red,wfdb.kBZ,exe_iQIBZ,Dmats[iQ_isymm],trev,wfdb.ktree)
    
    return AQ_rot
