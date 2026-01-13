##
## Authors: MN (FP adapted, SB optimized)
##
import numpy as np
import os
from yambopy.dbs.excitondb import YamboExcitonDB
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


def exciton_phonon_matelem(latdb,elphdb,wfdb,Qrange=[0,1],BSE_dir='bse',BSE_Lin_dir=None,
                           neigs=-1,dmat_mode='run',save_files=True,exph_file='Ex-ph.npy',overwrite=False):
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
    Qrange : int list, optional
        Exciton Qpoint index range [iQ_initial, iQ_final] (python counting). Default is [0,0] (Gamma point only).
        Note that the indexing is in full BZ and not in iBZ. See wfc.kBZ to see the kpoints in full BZ
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    dmat_mode : str, optional
        If 'save', print dmats on .npy file for faster recalculation. If 'load', load from .npy file. Else, calculate Dmats at runtime.
    save_files : bool, optional
        If True, the matrix elements will be saved in .npy file `exph_file`. Default is True.
    overwrite : bool, optional
        If False and `exph_file` is found, the matrix elements will be loaded from file. Default is False.
    """

    # Check if we just need to load
    if os.path.exists(exph_file) and overwrite==False:
        print(f'Loading EXCPH matrices from {exph_file}...')
        exph_mat_loaded = np.load(exph_file)
        return exph_mat_loaded

    # Load exc dbs
    exdbs = []
    for iQ in tqdm(range(Qrange[0],Qrange[1])):
        # wfdb.kpoints_indexes[iQ]
        filename = 'ndb.BS_diago_Q%d' % (latdb.kpoints_indexes[iQ]+1)
        excdb = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=BSE_dir,\
                                            Load_WF=True, neigs=neigs)
        exdbs.append(excdb)

    # get D matrices
    Dmats = save_or_load_dmat(wfdb,mode=dmat_mode,dmat_file='Dmats.npy')

    # Calculation
    print('Calculating EXCPH matrix elements...')
    exph_mat = []
    for iQ in tqdm(range(Qrange[0],Qrange[1])):
        Q_in = wfdb.kBZ[iQ]
        exph_mat.append( exciton_phonon_matelem_iQ(elphdb,wfdb,exdbs,Dmats,\
                                                   BSE_Lin_dir=BSE_Lin_dir,Q_in=Q_in,neigs=neigs) )
    # IO
    if len(exph_mat)<2: exph_mat = exph_mat[0] # single Q-point calculation (suppress axis)
    else:               exph_mat = np.array(exph_mat) #[nQ,nq,nmodes,nexc_in (Qexc),nexc_out (Qexc+q)]
    
    if save_files: 
        if exph_file[-4:]!='.npy': exph_file = exph_file+'.npy'
        print(f'Excph coupling file saved to {exph_file}')
        np.save(exph_file,exph_mat)
    
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
    for iq in range(elphdb.nq):
        ph_eig, elph_mat = elphdb.read_iq(iq,convention='standard')
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
    Qpt reduced coordinates in BZ or whatever
    '''
    latdb = wfdb.ydb
    idx_BZQ = wfdb.kptBZidx(Qpt)
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
