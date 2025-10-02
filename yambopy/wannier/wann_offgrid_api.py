import os
import numpy as np
from typing import Dict, Tuple

from .wann_bse_wannier import BSEWannierFT, kernel_realspace_roundtrip
from yambopy.dbs.bsekerneldb import YamboBSEKernelDB
from yambopy import YamboExcitonDB

# ------------------------------
# Helpers (centralized)
# ------------------------------

def _align_U_to_kpoints(tbmodel, kpoints: np.ndarray, *, decimals: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U_val, U_cond) from tbmodel aligned to given kpoints order using periodic lookup."""
    U_val_tb, U_cond_tb = tbmodel.get_U_val_cond()
    try:
        from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
        tb_k = np.asarray(tbmodel.mpgrid.k, float)
        tree = build_ktree(tb_k)
        idx_tb = np.empty(kpoints.shape[0], dtype=int)
        tol = max(1e-5, 0.5 * 10.0**(-decimals))
        for i, ky in enumerate(kpoints):
            idx_tb[i] = int(find_kpt(tree, make_kpositive(ky), tol=tol))
        return U_val_tb[idx_tb], U_cond_tb[idx_tb]
    except Exception:
        return U_val_tb, U_cond_tb


def _build_dual_from_k(kpts: np.ndarray, centered: bool, decimals: int = 10) -> np.ndarray:
    def _infer_grid_shape_from_k(kpts_: np.ndarray, decimals_: int = 10):
        k = np.mod(np.asarray(kpts_, float), 1.0)
        r = np.round(k, decimals=decimals_)
        xs = np.unique(r[:, 0]); ys = np.unique(r[:, 1]); zs = np.unique(r[:, 2])
        return len(xs), len(ys), len(zs)

    def _dual_indices(n: int, centered_: bool) -> np.ndarray:
        return np.arange(-(n // 2), n - (n // 2), dtype=int) if centered_ else np.arange(n, dtype=int)

    nkx, nky, nkz = _infer_grid_shape_from_k(kpts, decimals)
    ix = _dual_indices(nkx, centered); iy = _dual_indices(nky, centered); iz = _dual_indices(nkz, centered)
    grid = np.array([[x, y, z] for x in ix for y in iy for z in iz], dtype=int)
    return grid


# ------------------------------
# 1) Read all Q and build 7D kernel K7(Q,k,k',v,c,v',c')
# ------------------------------

def build_kernel7_from_yambo(lat_k, folder_bse: str, diag: bool = False) -> Dict:
    """
    Read all available ndb.BS_diago_Q* and ndb.BS_PAR_Q* files and assemble
    K7 with shape (NQ, Nk, Nk, Nv, Nc, Nv, Nc).
    Returns dict with keys: K7, kpoints, Qpoints, val_bands, cond_bands.

    If diag is True, print per-Q diagnostics about transition counts and band sets.
    """
    kpoints = np.asarray(lat_k.red_kpoints, dtype=float)
    # Discover Q files by scanning sequentially until a missing file
    Q_indices = []
    iq = 1
    while True:
        par_path = os.path.join(folder_bse, f'ndb.BS_PAR_Q{iq}')
        exc_path = os.path.join(folder_bse, f'ndb.BS_diago_Q{iq}')
        if os.path.isfile(par_path) and os.path.isfile(exc_path):
            Q_indices.append(iq)
            iq += 1
            continue
        break
    if not Q_indices:
        raise FileNotFoundError(f"No ndb.BS_PAR_Q* (and matching diago) found in {folder_bse}")

    if diag:
        print(f"[Diag] Found {len(Q_indices)} Q-points: {[int(q) for q in Q_indices]}")

    # Use Q1 to fix band sets
    bsk0 = YamboBSEKernelDB.from_db_file(lat_k, Qpt=Q_indices[0], folder=folder_bse)
    exc0 = YamboExcitonDB.from_db_file(lat_k, folder=folder_bse, filename=f'ndb.BS_diago_Q{Q_indices[0]}')
    _, val_bands, cond_bands = bsk0.as_band_kernel_6d(exc0)
    Nv, Nc = len(val_bands), len(cond_bands)
    Nk = kpoints.shape[0]
    NQ = len(Q_indices)

    if diag:
        # Report baseline band sets at Q1
        vb0 = np.array(sorted(np.unique(exc0.table[:, 1]).tolist()), dtype=int)
        cb0 = np.array(sorted(np.unique(exc0.table[:, 2]).tolist()), dtype=int)
        ntrans_table0 = int(exc0.ntransitions)
        ntrans_kernel0 = int(bsk0.ntransitions)
        print(f"[Diag][Q{Q_indices[0]}] ntrans_kernel={ntrans_kernel0}, ntrans_table={ntrans_table0}, nexcitons={exc0.nexcitons}")
        print(f"[Diag][Q{Q_indices[0]}] Nv={len(vb0)}, Nc={len(cb0)}, val_bands={vb0.tolist()}, cond_bands={cb0.tolist()}")
        if int(np.max(exc0.table[:, 0])) != Nk:
            print(f"[Diag][Q{Q_indices[0]}] WARNING: table nk={int(np.max(exc0.table[:, 0]))} differs from lattice Nk={Nk}")

    K7 = np.zeros((NQ, Nk, Nk, Nv, Nc, Nv, Nc), dtype=np.complex128)
    for iq0, Qpt in enumerate(Q_indices):
        exc_db = YamboExcitonDB.from_db_file(lat_k, folder=folder_bse, filename=f'ndb.BS_diago_Q{Qpt}')
        bsk_db = YamboBSEKernelDB.from_db_file(lat_k, Qpt=Qpt, folder=folder_bse)

        if diag:
            vb_q = np.array(sorted(np.unique(exc_db.table[:, 1]).tolist()), dtype=int)
            cb_q = np.array(sorted(np.unique(exc_db.table[:, 2]).tolist()), dtype=int)
            print(f"[Diag][Q{Qpt}] ntrans_kernel={int(bsk_db.ntransitions)}, ntrans_table={int(exc_db.ntransitions)}, nexcitons={exc_db.nexcitons}")
            print(f"[Diag][Q{Qpt}] Nv={len(vb_q)}, Nc={len(cb_q)}, val_bands={vb_q.tolist()}, cond_bands={cb_q.tolist()}")
            if int(np.max(exc_db.table[:, 0])) != Nk:
                print(f"[Diag][Q{Qpt}] WARNING: table nk={int(np.max(exc_db.table[:, 0]))} differs from lattice Nk={Nk}")

        K6, vb, cb = bsk_db.as_band_kernel_6d(exc_db)
        # Sanity: consistent band sets
        if not (np.array_equal(vb, val_bands) and np.array_equal(cb, cond_bands)):
            raise ValueError("Band sets vary across Q files; please ensure consistent Nv/Nc.")
        K7[iq0] = K6

    # Qpoints in reduced coords align with kpoints order (Yambo convention)
    Qpoints = kpoints.copy()
    return dict(K7=K7, kpoints=kpoints, Qpoints=Qpoints, val_bands=val_bands, cond_bands=cond_bands)


# ------------------------------
# 2) Build W once (Q-independent) and provide reconstruction utilities
# ------------------------------

def build_W_state_from_yambo(lat_k, tbmodel, folder_bse: str, *, centered: bool = False, decimals: int = 6,
                             U_bloch_to_wann: np.ndarray | None = None, use_Q_index: int = 0, diag: bool = False) -> Dict:
    """
    High-level pipeline:
    - Read all Q to build K7.
    - Choose U matrices (Wannier gauge if provided, else TB).
    - Build Sh/Se shells from k-grid and compute W(Sh,Se,R0) via a single-Q roundtrip.
    Returns a state dict suitable for on-grid and off-grid reconstructions.

    If diag is True, print per-Q diagnostics while assembling K7.
    """
    data = build_kernel7_from_yambo(lat_k, folder_bse, diag=diag)
    K7 = data['K7']
    kpoints = data['kpoints']
    val_bands = data['val_bands']
    cond_bands = data['cond_bands']

    # U source
    if U_bloch_to_wann is not None:
        from .wann_bse_wannier import split_U_bloch_to_wann
        U_val, U_cond = split_U_bloch_to_wann(U_bloch_to_wann, val_bands, cond_bands)
    else:
        U_val, U_cond = _align_U_to_kpoints(tbmodel, kpoints, decimals=decimals)

    # Validate TB partition if TB is used
    if U_bloch_to_wann is None:
        Nv_y = len(val_bands)
        Nc_y = len(cond_bands)
        if getattr(tbmodel, 'nv', None) != Nv_y or getattr(tbmodel, 'nc', None) != Nc_y:
            raise ValueError(
                f"TB (nv={getattr(tbmodel,'nv',None)}, nc={getattr(tbmodel,'nc',None)}) != Yambo (Nv={Nv_y}, Nc={Nc_y})."
            )

    # Shells
    Sh_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)
    Se_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)

    # Build W using a single Q slice (Q-independent result expected)
    iq_use = int(use_Q_index)
    if iq_use < 0 or iq_use >= K7.shape[0]:
        raise IndexError("use_Q_index out of range")
    K6_ref = K7[iq_use]

    # Use the actual Q of this slice for a consistent forward transform.
    # W ends up Q-independent, but the Bloch->Wtilde step must use the matching Q.
    W, K_back, R0_list = kernel_realspace_roundtrip(
        K6_ref,
        kpoints=kpoints,
        Q=data['Qpoints'][iq_use],
        U_val=U_val,
        U_cond=U_cond,
        Sh_list=Sh_list,
        Se_list=Se_list,
        centered=centered,
        decimals=decimals,
    )

    return dict(
        W=W,
        R0_list=R0_list,
        Sh_list=Sh_list,
        Se_list=Se_list,
        kpoints=kpoints,
        U_val=U_val,
        U_cond=U_cond,
        val_bands=val_bands,
        cond_bands=cond_bands,
        Qpoints=data['Qpoints'],
        decimals=decimals,
        # Optionally attach K7 for validation
        K7_ref=K7,
        # Diagnostics for the slice used to build W
        K6_ref_Q=K6_ref,
        K6_back_Q=K_back,
        iq_used=iq_use,
    )


def reconstruct_K7_on_grid(state: Dict) -> np.ndarray:
    """
    Reconstruct full K7(Q,k,k',v,c,v',c') on the original coarse q-grid using stored W and U.
    """
    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    kpoints = state['kpoints']
    Qpoints = state['Qpoints']
    U_val = state['U_val']
    U_cond = state['U_cond']
    decimals = int(state.get('decimals', 10))

    Nk = kpoints.shape[0]
    NQ = Qpoints.shape[0]

    # Precompute Wtilde for all on-grid q values using grid FT (exact inverse on the grid)
    ft0 = BSEWannierFT(kpoints=kpoints, Q=np.zeros(3, float), U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list, decimals=decimals)
    Wtilde_all_q = ft0.W_to_wtilde(W, R0_list)  # (NRh,NRe,Nk,Nm,Nm)

    # Build per-Q transformer and invert
    K7 = np.zeros((NQ, Nk, Nk, U_val.shape[2], U_cond.shape[2], U_val.shape[2], U_cond.shape[2]), dtype=np.complex128)
    for iqQ in range(NQ):
        Q_vec = Qpoints[iqQ]
        ft = BSEWannierFT(kpoints=kpoints, Q=Q_vec, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list, decimals=decimals)
        K6 = ft.wtilde_to_bloch(Wtilde_all_q)
        K7[iqQ] = K6
    return K7


# ------------------------------
# 3) Off-grid interpolation in q for given Q
# ------------------------------

def evaluate_K_for_qQ(state: Dict, tbmodel, q_vec, Q_vec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct K(k,k+q; Q) for arbitrary q (off-grid allowed) and given Q.
    Returns (K_per_k, kpq_indices) with K_per_k shape (Nk,Nv,Nc,Nv,Nc) and kpq as on-grid map for reference.
    """
    q = np.asarray(q_vec, float)
    Q = np.asarray(Q_vec, float)
    if q.shape != (3,) or Q.shape != (3,):
        raise ValueError("q_vec and Q_vec must have shape (3,)")

    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    kpoints = state['kpoints']
    decimals = int(state.get('decimals', 10))

    # Wtilde at this q (arbitrary)
    Wtilde_q = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W, R0_list, q_list=q[None, :])[:, :, 0, :, :]

    # Build U at k, k+q, k−Q, k+q−Q via TB HR interpolation
    Uvk, Uck = tbmodel.get_U_val_cond_at_kpoints(kpoints)
    Uvk_mQ, _ = tbmodel.get_U_val_cond_at_kpoints(np.mod(kpoints - Q, 1.0))
    Uvk_pq_mQ, Uck_pq = tbmodel.get_U_val_cond_at_kpoints(np.mod(kpoints + q - Q, 1.0))

    # Reference kpq indices on the coarse grid (best-effort lookup)
    try:
        from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
        Nk = kpoints.shape[0]
        tree = build_ktree(np.asarray(kpoints, float))
        tol = max(1e-5, 0.5 * 10.0**(-decimals))
        kpq = np.empty(Nk, dtype=int)
        for ik in range(Nk):
            kpq[ik] = int(find_kpt(tree, make_kpositive(kpoints[ik] + q), tol=tol))
    except Exception:
        r_k = np.round(np.mod(kpoints, 1.0), decimals=decimals)
        lut = {tuple(r_k[i]): i for i in range(r_k.shape[0])}
        Nk = kpoints.shape[0]
        kpq = np.empty(Nk, dtype=int)
        for ik in range(Nk):
            kpq[ik] = lut[tuple(np.round(np.mod(kpoints[ik] + q, 1.0), decimals=decimals))]

    # Reconstruct using general q,Q formula
    K_per_k = BSEWannierFT.reconstruct_bloch_for_qQ(
        Wtilde_q=Wtilde_q,
        kpoints=kpoints,
        Sh_list=Sh_list,
        Se_list=Se_list,
        q_vec=q,
        Q_vec=Q,
        Uv_k_minus_Q=Uvk_mQ,
        Uc_k=Uck,
        Uv_k_plus_q_minus_Q=Uvk_pq_mQ,
        Uc_k_plus_q=Uck_pq,
    )
    return K_per_k, kpq


# ------------------------------
# Backward-compat wrappers (Gamma-Q only)
# ------------------------------

def build_W_from_yambo_kernel(lat_k, tbmodel, bsk_db, excitons, *, centered: bool = False, decimals: int = 6,
                              U_bloch_to_wann: np.ndarray | None = None) -> Dict:
    """Deprecated: kept for compatibility. Prefer build_W_state_from_yambo."""
    # 1) Kernel with full val/cond sets from Yambo
    K6, val_bands, cond_bands = bsk_db.as_band_kernel_6d(excitons)

    # 2) k-points (Yambo order) and U matrices source
    kpoints = np.asarray(lat_k.red_kpoints, dtype=float)
    if U_bloch_to_wann is not None:
        from .wann_bse_wannier import split_U_bloch_to_wann
        U_val, U_cond = split_U_bloch_to_wann(U_bloch_to_wann, val_bands, cond_bands)
    else:
        U_val, U_cond = _align_U_to_kpoints(tbmodel, kpoints, decimals=decimals)

    # Shells
    Sh_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)
    Se_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)

    # Roundtrip at Q=0
    W, K_back, R0_list = kernel_realspace_roundtrip(
        K6,
        kpoints=kpoints,
        Q=np.zeros(3, float),
        U_val=U_val,
        U_cond=U_cond,
        Sh_list=Sh_list,
        Se_list=Se_list,
        centered=centered,
        decimals=decimals,
    )

    return dict(
        W=W,
        R0_list=R0_list,
        Sh_list=Sh_list,
        Se_list=Se_list,
        kpoints=kpoints,
        U_val=U_val,
        U_cond=U_cond,
        K6=K_back,
        val_bands=val_bands,
        cond_bands=cond_bands,
        decimals=decimals,
    )


def build_yambo_wtilde_cache(state: Dict, lat_k, folder_bse: str, q_indices=None) -> Dict[int, np.ndarray]:
    """Deprecated: not needed once K7 workflow is used."""
    kpoints = state['kpoints']
    Uv = state['U_val']
    Uc = state['U_cond']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    decimals = int(state.get('decimals', 10))

    Nk = kpoints.shape[0]
    if q_indices is None:
        q_indices = range(Nk)

    cache: Dict[int, np.ndarray] = {}
    for iq in q_indices:
        exc_db_q = YamboExcitonDB.from_db_file(lat_k, folder=folder_bse, filename=f'ndb.BS_diago_Q{iq+1}')
        kernel_db_q = YamboBSEKernelDB.from_db_file(lat_k, Qpt=iq+1, folder=folder_bse)
        K6_ref, _, _ = kernel_db_q.as_band_kernel_6d(exc_db_q)
        ft = BSEWannierFT(kpoints=kpoints, Q=kpoints[iq], U_val=Uv, U_cond=Uc, Sh_list=Sh_list, Se_list=Se_list, decimals=decimals)
        Wtilde_ref = ft.bloch_to_wtilde(K6_ref)
        cache[int(iq)] = Wtilde_ref[:, :, iq, :, :]
    return cache


def evaluate_offgrid_K_from_state(state: Dict, q_vec, tbmodel, k_ref=None, use_cache: bool = True):
    """Deprecated: prefer evaluate_K_for_qQ with Q_vec set to (0,0,0)."""
    q = np.asarray(q_vec, float)
    if q.shape != (3,):
        raise ValueError("q_vec must have shape (3,)")

    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    kpoints = state['kpoints'] if k_ref is None else np.asarray(k_ref, float)
    decimals = int(state.get('decimals', 10))

    # On-grid detection via rounded reduced coordinates
    r_k = np.round(np.mod(kpoints, 1.0), decimals=decimals)
    lut = {tuple(r_k[i]): i for i in range(r_k.shape[0])}
    rq = tuple(np.round(np.mod(q, 1.0), decimals=decimals))

    # Decide Wq source
    Wtilde_off = None
    if use_cache and rq in lut and 'Wq_cache' in state:
        iq = lut[rq]
        if iq in state['Wq_cache']:
            Wtilde_off = state['Wq_cache'][iq]
    if Wtilde_off is None:
        if rq in lut:
            iq = lut[rq]
            ft = BSEWannierFT(
                kpoints=kpoints,
                Q=np.zeros(3, float),
                U_val=state['U_val'],
                U_cond=state['U_cond'],
                Sh_list=Sh_list,
                Se_list=Se_list,
                decimals=decimals,
            )
            Wtilde_all = ft.W_to_wtilde(W, R0_list)
            Wtilde_off = Wtilde_all[:, :, iq, :, :]
        else:
            Wtilde_all = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W, R0_list, q_list=q[None, :])
            Wtilde_off = Wtilde_all[:, :, 0, :, :]

    # Build index maps and U matrices
    if rq in lut:
        try:
            from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
            Nk = kpoints.shape[0]
            tree = build_ktree(np.asarray(kpoints, float))
            tol = max(1e-5, 0.5 * 10.0**(-decimals))
            kmq = np.empty(Nk, dtype=int)
            kpq = np.empty(Nk, dtype=int)
            for ik in range(Nk):
                kmq[ik] = int(find_kpt(tree, make_kpositive(kpoints[ik] - q), tol=tol))
                kpq[ik] = int(find_kpt(tree, make_kpositive(kpoints[ik] + q), tol=tol))
        except Exception:
            Nk = kpoints.shape[0]
            kmq = np.empty(Nk, dtype=int)
            kpq = np.empty(Nk, dtype=int)
            for ik in range(Nk):
                kmq[ik] = lut[tuple(np.round(np.mod(kpoints[ik] - q, 1.0), decimals=decimals))]
                kpq[ik] = lut[tuple(np.round(np.mod(kpoints[ik] + q, 1.0), decimals=decimals))]

        Uv_k = state['U_val']
        Uc_k = state['U_cond']
        Uv_kmq = Uv_k[kmq]
        Uc_kpq = Uc_k[kpq]
    else:
        U_packs = tbmodel.get_U_for_offgrid_q(q, k_ref=kpoints)
        Uv_k, Uc_k = U_packs['k']
        Uv_kmq, _ = U_packs['k_minus_q']
        _, Uc_kpq = U_packs['k_plus_q']

        try:
            from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
            Nk = kpoints.shape[0]
            tree = build_ktree(np.asarray(kpoints, float))
            tol = max(1e-5, 0.5 * 10.0**(-decimals))
            kpq = np.empty(Nk, dtype=int)
            for ik in range(Nk):
                kpq[ik] = int(find_kpt(tree, make_kpositive(kpoints[ik] + q), tol=tol))
        except Exception:
            Nk = kpoints.shape[0]
            kpq = np.empty(Nk, dtype=int)
            for ik in range(Nk):
                kpq[ik] = lut[tuple(np.round(np.mod(kpoints[ik] + q, 1.0), decimals=decimals))]

    K_off = BSEWannierFT.reconstruct_bloch_for_offgrid_q(
        Wtilde_q=Wtilde_off,
        kpoints=kpoints,
        Sh_list=Sh_list,
        Se_list=Se_list,
        q_vec=q,
        Uv_k_minus_q=Uv_kmq,
        Uc_k=Uc_k,
        Uv_k=Uv_k,
        Uc_k_plus_q=Uc_kpq,
    )
    return K_off, kpq


def evaluate_offgrid_K6_all_pairs(state: Dict, *, Q_vec=None, k_ref=None):
    """
    Build the full 6D kernel K6(ik, ik', v, c, v', c') for all k–k' pairs using the stored
    real-space kernel W(Sh,Se,R0) and U matrices from the state.

    - If Q_vec is None, use Q = (0,0,0). Q enters only via L(k) = U_val(k−Q) ⊗ U_cond(k).
    - k_ref (Nk,3) can be provided to override k-point order; U matrices are realigned accordingly.

    Returns:
      - K6: (Nk, Nk, Nv, Nc, Nv, Nc)
    """
    import numpy as _np
    q_dtype = float

    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    k_state = _np.asarray(state['kpoints'], q_dtype)
    U_val_state = _np.asarray(state['U_val'], complex)
    U_cond_state = _np.asarray(state['U_cond'], complex)
    decimals = int(state.get('decimals', 10))

    kpoints = k_state if k_ref is None else _np.asarray(k_ref, q_dtype)
    if k_ref is not None:
        try:
            from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
            tree = build_ktree(k_state)
            Nk = kpoints.shape[0]
            perm = _np.empty(Nk, dtype=int)
            tol = max(1e-5, 0.5 * 10.0**(-decimals))
            for i in range(Nk):
                perm[i] = int(find_kpt(tree, make_kpositive(kpoints[i]), tol=tol))
            U_val = U_val_state[perm]
            U_cond = U_cond_state[perm]
        except Exception:
            U_val = U_val_state
            U_cond = U_cond_state
    else:
        U_val = U_val_state
        U_cond = U_cond_state

    Q = _np.zeros(3, q_dtype) if Q_vec is None else _np.asarray(Q_vec, q_dtype)

    Wtilde_all = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W, R0_list, q_list=kpoints)

    ft = BSEWannierFT(
        kpoints=kpoints,
        Q=Q,
        U_val=U_val,
        U_cond=U_cond,
        Sh_list=Sh_list,
        Se_list=Se_list,
        decimals=decimals,
    )
    K6 = ft.wtilde_to_bloch(Wtilde_all)
    return K6