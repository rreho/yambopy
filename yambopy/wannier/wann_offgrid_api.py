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

    # 2) k-points and U matrices from TB model
    kpoints = np.asarray(lat_k.red_kpoints, dtype=float)
    U_val, U_cond = tbmodel.get_U_val_cond()

    # 3) Validate that TB partition matches Yambo's full Nv/Nc
    Nv_y = len(val_bands)
    Nc_y = len(cond_bands)
    if getattr(tbmodel, 'nv', None) != Nv_y or getattr(tbmodel, 'nc', None) != Nc_y:
        raise ValueError(
            f"TB model (nv={getattr(tbmodel,'nv',None)}, nc={getattr(tbmodel,'nc',None)}) does not match Yambo bands (Nv={Nv_y}, Nc={Nc_y}).\n"
            "Ensure the Wannier TB window and Fermi level produce the same valence/conduction counts as the Yambo kernel."
        )

    # 4) Dual shells (unsymmetrized) for exact inversion
    Sh_list, Se_list = tbmodel.build_shells_for_bse(mode='dual', symmetric=False)

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
    )


def evaluate_offgrid_K_from_state(state: Dict, q_vec, tbmodel, k_ref=None):
    """
    Given a state from build_W_from_yambo_kernel and an arbitrary reduced q_vec (shape (3,)),
    reconstruct K(k, k+q_vec) over the reference k grid.
    Optionally provide k_ref (Nk,3) to override the state's kpoints.
    Returns array (Nk, Nv, Nc, Nv, Nc).
    """
    q = np.asarray(q_vec, float)
    if q.shape != (3,):
        raise ValueError("q_vec must have shape (3,)")

    W = state['W']
    R0_list = state['R0_list']
    Sh_list = state['Sh_list']
    Se_list = state['Se_list']
    kpoints = state['kpoints'] if k_ref is None else np.asarray(k_ref, float)

    # 1) Evaluate Wtilde at off-grid q (select q-dimension index 0)
    Wtilde_all = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W, R0_list, q_list=q[None, :])
    Wtilde_off = Wtilde_all[:, :, 0, :, :]  # (NRh,NRe,Nm,Nm)

    # 2) Build U at k, k−q, k+q via HR interpolation
    U_packs = tbmodel.get_U_for_offgrid_q(q, k_ref=kpoints)
    Uv_k, Uc_k = U_packs['k']
    Uv_kmq, _ = U_packs['k_minus_q']
    _, Uc_kpq = U_packs['k_plus_q']

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
    return K_off