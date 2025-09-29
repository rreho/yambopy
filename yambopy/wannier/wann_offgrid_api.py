import numpy as np
from typing import Dict

from .wann_bse_wannier import BSEWannierFT, kernel_realspace_roundtrip


def build_W_from_yambo_kernel(lat_k, tbmodel, bsk_db, excitons, *, centered: bool = False, decimals: int = 6) -> Dict:
    """
    Build W(Sh,Se,R0) once from a Yambo kernel (full Nv,Nc) and TB U matrices, using dual shells.
    Returns a state dict with everything needed to evaluate off-grid q later.
    """
    # 1) Kernel with full val/cond sets from Yambo
    K6, val_bands, cond_bands = bsk_db.as_band_kernel_6d(excitons)

    # 2) k-points (Yambo order) and U matrices from TB model (TB order)
    kpoints = np.asarray(lat_k.red_kpoints, dtype=float)
    U_val_tb, U_cond_tb = tbmodel.get_U_val_cond()

    # Align TB U matrices to the Yambo k-point order to ensure consistent indexing
    try:
        from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
        tb_k = np.asarray(tbmodel.mpgrid.k, float)
        tree = build_ktree(tb_k)
        # Map each Yambo k to its index in TB grid using periodic lookup
        idx_tb = np.empty(kpoints.shape[0], dtype=int)
        # Tolerance consistent with decimals
        tol = max(1e-5, 0.5 * 10.0**(-decimals))
        for i, ky in enumerate(kpoints):
            idx_tb[i] = int(find_kpt(tree, make_kpositive(ky), tol=tol))
        U_val = U_val_tb[idx_tb]
        U_cond = U_cond_tb[idx_tb]
    except Exception:
        # Fallback: assume identical ordering
        U_val, U_cond = U_val_tb, U_cond_tb

    # 3) Validate that TB partition matches Yambo's full Nv/Nc
    Nv_y = len(val_bands)
    Nc_y = len(cond_bands)
    if getattr(tbmodel, 'nv', None) != Nv_y or getattr(tbmodel, 'nc', None) != Nc_y:
        raise ValueError(
            f"TB model (nv={getattr(tbmodel,'nv',None)}, nc={getattr(tbmodel,'nc',None)}) does not match Yambo bands (Nv={Nv_y}, Nc={Nc_y}).\n"
            "Ensure the Wannier TB window and Fermi level produce the same valence/conduction counts as the Yambo kernel."
        )

    # 4) Dual shells (unsymmetrized) for exact inversion
    #    Prefer full dual sets inferred from the k-grid to ensure perfect inversion.
    def _infer_grid_shape_from_k(kpts: np.ndarray, decimals: int = 10):
        k = np.mod(np.asarray(kpts, float), 1.0)
        r = np.round(k, decimals=decimals)
        xs = np.unique(r[:, 0]); ys = np.unique(r[:, 1]); zs = np.unique(r[:, 2])
        return len(xs), len(ys), len(zs)

    def _dual_indices(n: int, centered: bool) -> np.ndarray:
        return np.arange(-(n // 2), n - (n // 2), dtype=int) if centered else np.arange(n, dtype=int)

    def _build_dual_from_k(kpts: np.ndarray, centered: bool, decimals: int = 10) -> np.ndarray:
        nkx, nky, nkz = _infer_grid_shape_from_k(kpts, decimals)
        ix = _dual_indices(nkx, centered); iy = _dual_indices(nky, centered); iz = _dual_indices(nkz, centered)
        grid = np.array([[x, y, z] for x in ix for y in iy for z in iz], dtype=int)
        return grid

    Sh_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)
    Se_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)

    # 5) Real-space kernel and R0 dual set (Q from DB is Gamma for Coulomb kernel)
    Q = np.zeros(3, float)
    W, K_back, R0_list = kernel_realspace_roundtrip(
        K6,
        kpoints=kpoints,
        Q=Q,
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


def evaluate_offgrid_K_from_state(state: Dict, q_vec, tbmodel, k_ref=None):
    """
    Given a state from build_W_from_yambo_kernel and an arbitrary reduced q_vec (shape (3,)),
    reconstruct K(k, k+q_vec) over the reference k grid.
    Optionally provide k_ref (Nk,3) to override the state's kpoints.
    Returns (K_per_k, kpq_used) where K_per_k has shape (Nk, Nv, Nc, Nv, Nc) and kpq_used maps k -> k+q.
    """
    q = np.asarray(q_vec, float)
    if q.shape != (3,):
        raise ValueError("q_vec must have shape (3,)")

    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    kpoints = state['kpoints'] if k_ref is None else np.asarray(k_ref, float)
    decimals = int(state.get('decimals', 10))

    # 1) Evaluate Wtilde at requested q (off-grid evaluation is supported)
    Wtilde_all = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W, R0_list, q_list=q[None, :])
    Wtilde_off = Wtilde_all[:, :, 0, :, :]  # (NRh,NRe,Nm,Nm)

    # 2) If q lies on the original grid, reuse stored U and periodic mappings to avoid gauge drift
    #    Otherwise, fall back to HR interpolation provided by the TB model.
    #    On-grid detection via rounded reduced coordinates at the configured precision.
    r_k = np.round(np.mod(kpoints, 1.0), decimals=decimals)
    lut = {tuple(r_k[i]): i for i in range(r_k.shape[0])}
    rq = tuple(np.round(np.mod(q, 1.0), decimals=decimals))

    if rq in lut:
        # On-grid path: build k±q index maps with the same periodic KDTree logic
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
            # Fallback using hash lookup (requires exact rounding match)
            Nk = kpoints.shape[0]
            kmq = np.empty(Nk, dtype=int)
            kpq = np.empty(Nk, dtype=int)
            for ik in range(Nk):
                kmq[ik] = lut[tuple(np.round(np.mod(kpoints[ik] - q, 1.0), decimals=decimals))]
                kpq[ik] = lut[tuple(np.round(np.mod(kpoints[ik] + q, 1.0), decimals=decimals))]

        # Use the same U used to build W
        Uv_k = state['U_val']
        Uc_k = state['U_cond']
        Uv_kmq = Uv_k[kmq]
        Uc_kpq = Uc_k[kpq]
    else:
        # Off-grid path: build U via HR interpolation (may differ by gauge from stored U)
        U_packs = tbmodel.get_U_for_offgrid_q(q, k_ref=kpoints)
        Uv_k, Uc_k = U_packs['k']
        Uv_kmq, _ = U_packs['k_minus_q']
        _, Uc_kpq = U_packs['k_plus_q']

        # Also build kpq using periodic mapping so caller can compare blocks correctly
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

    # 3) Reconstruct K(k,k+q)
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

    # Inputs from state
    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    k_state = _np.asarray(state['kpoints'], q_dtype)
    U_val_state = _np.asarray(state['U_val'], complex)
    U_cond_state = _np.asarray(state['U_cond'], complex)
    decimals = int(state.get('decimals', 10))

    # Resolve kpoints (optionally override) and realign U if needed
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
            # Fallback: assume identical order
            U_val = U_val_state
            U_cond = U_cond_state
    else:
        U_val = U_val_state
        U_cond = U_cond_state

    # Q vector (default Gamma)
    Q = _np.zeros(3, q_dtype) if Q_vec is None else _np.asarray(Q_vec, q_dtype)

    # 1) Evaluate Wtilde for all on-grid q = kpoints (covers all k' via k' = k + q)
    Wtilde_all = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W, R0_list, q_list=kpoints)

    # 2) Use the inverse transform to assemble full K6 over all k,k'
    ft = BSEWannierFT(
        kpoints=kpoints,
        Q=Q,
        U_val=U_val,
        U_cond=U_cond,
        Sh_list=Sh_list,
        Se_list=Se_list,
        decimals=decimals,
    )
    K6 = ft.wtilde_to_bloch(Wtilde_all)  # (Nk,Nk,Nv,Nc,Nv,Nc)
    return K6