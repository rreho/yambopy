import numpy as np
from typing import Tuple, Optional

from .wann_bse_wannier import build_k_minus_vector_indices
from yambopy.units import ha2ev

def _align_tb_to_kpoints(tb_k: np.ndarray, kpoints: np.ndarray, decimals: int = 6) -> np.ndarray:
    """
    Return indices idx_tb such that tb_k[idx_tb[i]] ~= kpoints[i] under periodic equivalence.
    """
    from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
    tree = build_ktree(np.asarray(tb_k, float))
    Nk = kpoints.shape[0]
    idx_tb = np.empty(Nk, dtype=int)
    tol = max(1e-5, 0.5 * 10.0**(-decimals))
    for i in range(Nk):
        idx_tb[i] = int(find_kpt(tree, make_kpositive(kpoints[i]), tol=tol))
    return idx_tb


def build_bse_resonant_hamiltonian(
    *,
    K6: np.ndarray,                     # (Nk,Nk,Nv,Nc,Nv,Nc)
    tbmodel,                            # TBMODEL with mpgrid, eigv, nv, nc (used if no external energies)
    kpoints: np.ndarray,                # (Nk,3) reduced, Yambo order
    Q_vec: np.ndarray,                  # (3,) reduced Q (e.g., Gamma for Coulomb)
    f_kn: Optional[np.ndarray] = None,  # (Nk, Nb) occupations; if None and TB path used: T=0 (val=1, cond=0)
    decimals: int = 6,
    kernel_scale: float = ha2ev,          # unit/sign factor for kernel (e.g., -HA2EV)
    include_antiresonant: bool = False, # if True, build 2Nt x 2Nt with antiresonant block
    include_coupling: bool = False,     # if True, include resonant–antiresonant coupling block B
    non_hermitian_blocks: bool = True,  # if True, use [[A, B], [-B*, -A*]]; else [[A, B],[B*, A*]]
    coupling_scale: float = 1.0,        # additional scale for B block (often 1.0)
    apply_occ_prefactor: bool = False,  # if True, scale rows by (f_c - f_v) like TDDFT forms
    # --- New optional sources for energies (decouple from TBMODEL) ---
    ev_kmQ: Optional[np.ndarray] = None,   # (Nk, Nv) valence energies at k-Q (eV)
    ec_k: Optional[np.ndarray] = None,     # (Nk, Nc) conduction energies at k (eV)
    energies_file: Optional[str] = None,   # npz with keys 'ev_kmQ','ec_k'
    electrons_folder: Optional[str] = None,# path containing ns.db1 (SAVE or BSE folder)
    val_bands: Optional[np.ndarray] = None,# 1-based Yambo band indices for valence used in K6
    cond_bands: Optional[np.ndarray] = None# 1-based Yambo band indices for conduction used in K6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the BSE Hamiltonian in the (k,v,c) transition basis.

    Default (resonant/TDA):
      - A = diag(ΔE) + K, with ΔE(k,v,c) = Ec(k,c) - Ev(k−Q,v)
      - K from K6 (Nk,Nk,Nv,Nc,Nv,Nc) flattened to (Nt,Nt)
      - Optional row scaling by (f_c(k) - f_v(k−Q)) when apply_occ_prefactor=True

    Energies sources:
      - External: (ev_kmQ, ec_k) arrays or an npz file with those keys
      - Electrons DB: set electrons_folder and pass val_bands/cond_bands (1-based)
      - Fallback: TBMODEL eigenvalues (aligned to kpoints) and tbmodel.nv/nc

    Returns:
      - H: (Nt,Nt) if resonant only; otherwise (2Nt,2Nt) complex128
      - T_table: (Nt, 3) int array mapping t -> (ik, iv, ic) for the resonant sector
    """
    kpoints = np.asarray(kpoints, float)
    Q = np.asarray(Q_vec, float)
    Nk, _, Nv, Nc, Nv2, Nc2 = K6.shape
    assert Nv == Nv2 and Nc == Nc2, "Inconsistent Nv/Nc in K6"

    # --- Determine energy source ---
    use_external = False
    Ev_kmQ = None
    Ec_k = None

    # 1) Load from npz file if provided
    if energies_file is not None:
        data = np.load(energies_file)
        Ev_kmQ = np.asarray(data['ev_kmQ'], float)
        Ec_k = np.asarray(data['ec_k'], float)
        use_external = True

    # 2) Direct arrays if provided
    if (ev_kmQ is not None) and (ec_k is not None):
        Ev_kmQ = np.asarray(ev_kmQ, float)
        Ec_k = np.asarray(ec_k, float)
        use_external = True

    # 3) Build from YamboElectronsDB if requested
    if (not use_external) and (electrons_folder is not None):
        if (val_bands is None) or (cond_bands is None):
            raise ValueError("When electrons_folder is set, val_bands and cond_bands (1-based) must be provided.")
        from yambopy.dbs.electronsdb import YamboElectronsDB
        from yambopy.lattice import car_red
        from yambopy.kpoints import build_ktree, find_kpt, make_kpositive
        # Load electrons DB (eigenvalues in eV), expand to full BZ
        edb = YamboElectronsDB.from_db_file(folder=electrons_folder, Expand=False)
        k_full_cart, _, _ = edb.expand_kpts()
        edb.expandEigenvalues()  # builds edb.eigenvalues with BZ duplication
        # Map full-BZ electrons k-points (to reduced) into provided reduced kpoints order
        k_full_red = car_red(k_full_cart, edb.rlat)
        tree = build_ktree(k_full_red)
        perm = np.array([int(find_kpt(tree, make_kpositive(k))) for k in kpoints], dtype=int)
        # Select band sets (convert 1-based Yambo to 0-based)
        v_idx = np.asarray(val_bands, dtype=int) - 1
        c_idx = np.asarray(cond_bands, dtype=int) - 1
        if len(v_idx) != Nv or len(c_idx) != Nc:
            raise ValueError(f"val_bands/cond_bands lengths ({len(v_idx)},{len(c_idx)}) do not match K6 Nv/Nc ({Nv},{Nc})")
        # electrons eigenvalues shape: [spin, Nk, Nb]; take spin=0
        E_full = edb.eigenvalues[0][perm, :]
        Ev_k = E_full[:, v_idx]  # (Nk, Nv)
        Ec_k = E_full[:, c_idx]  # (Nk, Nc)
        # Build k-Q mapping and Ev(k-Q)
        kmQ = build_k_minus_vector_indices(kpoints, Q, decimals=decimals)
        Ev_kmQ = Ev_k[kmQ, :]
        use_external = True

    # --- Fallback: TBMODEL eigenvalues path ---
    if not use_external:
        # Align TB eigenvalues to provided kpoints order
        if not hasattr(tbmodel, 'mpgrid'):
            raise ValueError("tbmodel.mpgrid is required to align eigenvalues to kpoints")
        tb_k = np.asarray(tbmodel.mpgrid.k, float)
        idx_tb = _align_tb_to_kpoints(tb_k, kpoints, decimals=decimals)
        # Gather eigenvalues (Nk, Nb) in the same order as kpoints
        if hasattr(tbmodel, 'eigv'):
            eigv = np.asarray(tbmodel.eigv)[idx_tb]  # (Nk, Nb)
        else:
            evals, _ = tbmodel.get_eigenval_and_vec(kpoints, from_hr=True)
            eigv = evals
        nv = int(getattr(tbmodel, 'nv'))
        nc = int(getattr(tbmodel, 'nc'))
        if not (nv == Nv and nc == Nc):
            raise ValueError(f"TB nv/nc ({nv},{nc}) do not match K6 ({Nv},{Nc})")
        # Energies per transition: Ev(k−Q,v), Ec(k,c)
        kmQ = build_k_minus_vector_indices(kpoints, Q, decimals=decimals)  # (Nk,)
        Ev_kmQ = eigv[kmQ, :nv]               # (Nk, Nv)
        Ec_k = eigv[:, nv:nv+nc]              # (Nk, Nc)
        eigv_for_occ = eigv
        nv_for_occ = nv
        nc_for_occ = nc
    else:
        # External energies path; we won't rely on TBMODEL for occupations
        eigv_for_occ = None
        nv_for_occ = Nv
        nc_for_occ = Nc

    # Sanity checks on external energies
    if Ev_kmQ is None or Ec_k is None:
        raise RuntimeError("Failed to construct external or TB energies for BSE Hamiltonian.")
    if Ev_kmQ.shape != (Nk, Nv) or Ec_k.shape != (Nk, Nc):
        raise ValueError(f"Energy shapes mismatch: Ev_kmQ {Ev_kmQ.shape}, Ec_k {Ec_k.shape}, expected {(Nk,Nv)} and {(Nk,Nc)}")

    # Diagonal ΔE(k,v,c)
    dE = Ec_k[:, None, :] - Ev_kmQ[:, :, None]  # (Nk, Nv, Nc)

    # Occupations prefactor (optional)
    if apply_occ_prefactor:
        if eigv_for_occ is None:
            # External energies: assume T=0, valence filled, conduction empty -> (f_c - f_v) = -1
            occ_diff = -np.ones((Nk, Nv, Nc), dtype=float)
        else:
            if f_kn is None:
                Nb = eigv_for_occ.shape[1]
                f = np.zeros((Nk, Nb), float)
                f[:, :nv_for_occ] = 1.0
            else:
                f = np.asarray(f_kn, float)
                if f.shape != eigv_for_occ.shape:
                    raise ValueError(f"f_kn shape {f.shape} must match eigv shape {eigv_for_occ.shape}")
            kmQ = build_k_minus_vector_indices(kpoints, Q, decimals=decimals)  # ensure exists
            occ_diff = f[:, nv_for_occ:nv_for_occ+nc_for_occ] - f[kmQ, :nv_for_occ]
            occ_diff = np.broadcast_to(occ_diff[:, None, :], (Nk, Nv, Nc))
    else:
        # Not used; placeholder to keep shapes consistent
        occ_diff = np.ones((Nk, Nv, Nc), dtype=float)

    # Build transition index table T_table and flatten helpers
    Nt = Nk * Nv * Nc
    T_table = np.zeros((Nt, 3), dtype=int)
    idx = 0
    for ik in range(Nk):
        for iv in range(Nv):
            for ic in range(Nc):
                T_table[idx] = (ik, iv, ic)
                idx += 1

    # Reshape K6 to (Nt,Nt)
    Ktt = K6.reshape(Nk, Nk, Nv*Nc, Nv*Nc)
    Ktt = Ktt.transpose(0, 2, 1, 3).reshape(Nt, Nt)

    # Row-wise occupation scaling (optional) and unit/sign factor
    row_occ = occ_diff.reshape(Nt)
    if apply_occ_prefactor:
        K_rowscaled = (row_occ[:, None]) * Ktt
    else:
        K_rowscaled = Ktt
    K_scaled = kernel_scale * K_rowscaled

    # Diagonal ΔE and resonant block A
    dE_flat = dE.reshape(Nt)
    A = np.diag(dE_flat) + K_scaled

    if not include_antiresonant:
        # Resonant (TDA) only
        return A.astype(np.complex128, copy=False), T_table

    # Build antiresonant and coupling blocks
    if include_coupling:
        B = coupling_scale * K_scaled
    else:
        B = np.zeros_like(A)

    if non_hermitian_blocks:
        # Casida-like non-Hermitian structure
        upper = np.concatenate([A, B], axis=1)
        lower = np.concatenate([-B.conj(), -A.conj()], axis=1)
        H = np.concatenate([upper, lower], axis=0)
    else:
        # Hermitian block structure sometimes used by alternative solvers
        upper = np.concatenate([A, B], axis=1)
        lower = np.concatenate([B.conj(), A.conj()], axis=1)
        H = np.concatenate([upper, lower], axis=0)

    return H.astype(np.complex128, copy=False), T_table