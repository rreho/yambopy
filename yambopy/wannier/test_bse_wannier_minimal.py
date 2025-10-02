#!/usr/bin/env python3
"""
Minimal diagnostic: verify single (k,k') block at Q!=Gamma using explicit formulas.
- Builds W from Gamma kernel once (state)
- Picks a target Q and a single ik, ik' = kpq[ik, iq]
- Computes K_explicit(ik,ik') via your equations using state.Wtilde and U
- Compares to Yambo K6 at the same Q, same (ik,ik') and band order
- Also performs a Γ control: reconstruct full K6 at Q=Γ from W and compare to Yambo.
- Prints Yambo's (k,k') mapping via excitons.table for the chosen off-Γ pair.

Run: python -m yambopy.wannier.test_bse_wannier_minimal
"""
import numpy as np
from numpy.linalg import norm

from yambopy import YamboLatticeDB, YamboExcitonDB
from yambopy.dbs.bsekerneldb import YamboBSEKernelDB
from yambopy.wannier.wann_model import TBMODEL
from yambopy.wannier.wann_io import HR as HR_reader
from yambopy.wannier.wann_nnkpgrid import NNKP_Grids
from yambopy.wannier.wann_offgrid_api import build_W_from_yambo_kernel, evaluate_offgrid_K_from_state, build_yambo_wtilde_cache
from yambopy.wannier.wann_bse_wannier import BSEWannierFT, kernel_realspace_roundtrip

from yambopy.lattice import car_red

def compute_single_block_explicit(state, q_vec, ik, ikp):
    """Compute K(ik,ikp) in band basis using explicit sums:
       K(ik,ikp;Q) = L(ik)^† [ (Nk/(NRh*NRe)) sum_{Sh,Se} e^{-i2π[(k−Q)·Sh + k·Se]} Wtilde(Sh,Se,q=ikp-ik) ] L(ikp)
    """
    # Unpack
    W = state['W']
    R0_list = state['R0_list']
    Sh = state['Sh_list']
    Se = state['Se_list']
    k = state['kpoints']
    Uv = state['U_val']
    Uc = state['U_cond']

    Nk = k.shape[0]
    NRh = Sh.shape[0]
    NRe = Se.shape[0]
    Nv = Uv.shape[2]
    Nc = Uc.shape[2]

    # Build transformer to get Wtilde for this specific q = k[ikp]-k[ik]
    ft = BSEWannierFT(kpoints=k, Q=q_vec, U_val=Uv, U_cond=Uc, Sh_list=Sh, Se_list=Se, decimals=state.get('decimals', 10))

    # Use kpq mapping: find iq such that k' = k + q, i.e., ft.kpq[ik, iq] == ikp
    iq_candidates = np.where(ft.kpq[ik] == ikp)[0]
    assert iq_candidates.size == 1, "Ambiguous or missing iq for given (ik,ikp)"
    iq = int(iq_candidates[0])

    # Compute Wtilde (full grid) and extract slab at iq
    Wtilde = ft.W_to_wtilde(W, R0_list)  # (NRh,NRe,Nk,Nm,Nm)
    Wq = Wtilde[:, :, iq, :, :]

    return compute_single_block_from_Wq(state, q_vec, ik, ikp, Wq)


def compute_single_block_from_Wq(state, q_vec, ik, ikp, Wq):
    """Compute K(ik,ikp) using a provided Wq slab (NRh,NRe,Nm,Nm) with standard phases and L."""
    # Unpack
    Sh = state['Sh_list']
    Se = state['Se_list']
    k = state['kpoints']
    Uv = state['U_val']
    Uc = state['U_cond']

    Nk = k.shape[0]
    NRh = Sh.shape[0]
    NRe = Se.shape[0]
    Nv = Uv.shape[2]
    Nc = Uc.shape[2]

    # Phases and norm (inverse uses complex-conjugate of forward): exp(+i2π[(k−Q)·Sh + k·Se])
    phases = np.exp(+2j * np.pi * ((k[ik] - q_vec) @ Sh.T))[:, None] * np.exp(+2j * np.pi * (k[ik] @ Se.T))[None, :]
    scale = Nk / float(NRh * NRe)

    # Sum over shells to get 𝓦_block in Wannier-me basis
    W_block = scale * np.einsum('rs,rsnm->nm', phases, Wq, optimize=True)

    # Build L(ik)=U_val(k−Q)⊗U_cond(k) and L(ikp)=U_val(k'−Q)⊗U_cond(k')
    ft_tmp = BSEWannierFT(kpoints=k, Q=q_vec, U_val=Uv, U_cond=Uc, Sh_list=Sh, Se_list=Se, decimals=state.get('decimals', 10))
    ikmQ = ft_tmp.kmQ[ik]
    ikpmQ = ft_tmp.kmQ[ikp]
    tens_k = np.einsum('mv,nc->mnvc', Uv[ikmQ], Uc[ik], optimize=True)
    Lk = tens_k.reshape(Uv.shape[1]*Uc.shape[1], Nv*Nc)
    tens_kp = np.einsum('mv,nc->mnvc', Uv[ikpmQ], Uc[ikp], optimize=True)
    Lkp = tens_kp.reshape(Uv.shape[1]*Uc.shape[1], Nv*Nc)

    # Project back to band basis
    Ktc = Lk.conj().T @ W_block @ Lkp
    return Ktc.reshape(Nv, Nc, Nv, Nc)


def stats(A, B):
    d = A - B
    max_abs = np.max(np.abs(d))
    mean_abs = np.mean(np.abs(d))
    denom = max(1e-16, np.max(np.abs(B)))
    rel_max = max_abs / denom
    return max_abs, mean_abs, rel_max


def phase_amplitude_drift(A, B, tol=1e-10):
    """Return (mean|ratio|, std|ratio|, mean phase, std phase) for element-wise ratio A/B on entries with |B|>tol."""
    mask = np.abs(B) > tol
    if not np.any(mask):
        return 0.0, 0.0, 0.0, 0.0
    R = A[mask] / B[mask]
    amps = np.abs(R)
    phs = np.angle(R)
    return float(np.mean(amps)), float(np.std(amps)), float(np.mean(phs)), float(np.std(phs))


def gamma_control(lat_k, state, folder_bse):
    """Reconstruct full K6 at Γ from W and compare to Yambo reference."""
    q_vec_g = np.zeros(3)
    ft_g = BSEWannierFT(
        kpoints=state['kpoints'], Q=q_vec_g,
        U_val=state['U_val'], U_cond=state['U_cond'],
        Sh_list=state['Sh_list'], Se_list=state['Se_list'],
        decimals=state.get('decimals', 10)
    )
    Wtilde_g = ft_g.W_to_wtilde(state['W'], state['R0_list'])
    K6_from_wtilde_g = ft_g.wtilde_to_bloch(Wtilde_g)

    exc_db_g = YamboExcitonDB.from_db_file(lat_k, folder=folder_bse, filename='ndb.BS_diago_Q1')
    kernel_db_g = YamboBSEKernelDB.from_db_file(lat_k, Qpt=1, folder=folder_bse)
    K6_ref_g, vbg, cbg = kernel_db_g.as_band_kernel_6d(exc_db_g)

    # Band sets should match state
    assert np.all(vbg == state['val_bands']) and np.all(cbg == state['cond_bands'])

    max_g, mean_g, rel_g = stats(K6_from_wtilde_g, K6_ref_g)
    print(f"[Gamma control] Full K6: Max |diff|={max_g:.3e}, Mean |diff|={mean_g:.3e}, Rel max={rel_g:.3e}")


def report_yambo_block_mapping(exc_db, ik, ikp, max_print=6):
    """Print Yambo excitons.table rows contributing to (k=ik,k'=ikp) block.
    Shows a few (k,v,c) entries for each side to verify ordering.
    """
    table = exc_db.table  # shape (Nt, >=3), 1-based in k,v,c
    t_left = np.where(table[:, 0] == ik + 1)[0]
    t_right = np.where(table[:, 0] == ikp + 1)[0]
    print(f"[Yambo mapping] #transitions at k={ik}: {t_left.size}; at k'={ikp}: {t_right.size}")
    if t_left.size:
        samp = t_left[:max_print]
        print("  Left k rows (first few):", [tuple(map(int, table[i, :3])) for i in samp])
    if t_right.size:
        samp = t_right[:max_print]
        print("  Right k' rows (first few):", [tuple(map(int, table[i, :3])) for i in samp])


def block_from_triangular(kernel_db, excitons, ik, ikp, val_bands, cond_bands):
    """Slice kernel[t_left][:, t_right] by transition indices and map to 4D (v,c,v',c')."""
    table = excitons.table
    ker = kernel_db.kernel
    t_left = np.where(table[:, 0] == ik + 1)[0]
    t_right = np.where(table[:, 0] == ikp + 1)[0]
    Nv = len(val_bands); Nc = len(cond_bands)
    vmap = {int(b): i for i, b in enumerate(val_bands)}
    cmap = {int(b): i for i, b in enumerate(cond_bands)}
    K = np.zeros((Nv, Nc, Nv, Nc), dtype=np.complex128)
    for t1 in t_left:
        v1 = vmap[int(table[t1, 1])]
        c1 = cmap[int(table[t1, 2])]
        for t2 in t_right:
            v2 = vmap[int(table[t2, 1])]
            c2 = cmap[int(table[t2, 2])]
            K[v1, c1, v2, c2] = ker[t1, t2]
    return K


def axis_permutation_diffs(A, B):
    """Compare A to B under axis permutations to detect v/v' or c/c' misalignments."""
    perms = {
        'id': A,
        'swap_v': np.transpose(A, (2, 1, 0, 3)),   # v<->v'
        'swap_c': np.transpose(A, (0, 3, 2, 1)),   # c<->c'
        'swap_vc': np.transpose(A, (2, 3, 0, 1)),  # both
    }
    out = {}
    for name, P in perms.items():
        out[name] = stats(P, B)
    return out


def compute_block_via_neg_q(state, q_vec, ik, ikp):
    """Compute block assuming opposite q-sign convention: use (-q, ikp, ik) and Hermitian swap."""
    K_rev = compute_single_block_explicit(state, -q_vec, ikp, ik)  # (Nv,Nc,Nv,Nc)
    # Hermitian relation: K(k,k') ~ [K(k',k)]^† with (vc) <-> (v'c')
    return np.conjugate(np.transpose(K_rev, (2, 3, 0, 1)))


def compute_single_block_explicit_alt(state, q_vec, ik, ikp):
    """Alternative convention test:
    Use L(ik)=U_val(k)⊗U_cond(k), L(ikp)=U_val(k')⊗U_cond(k'), and phases exp(+i2π[k·Sh + (k+Q)·Se]).
    """
    # Unpack
    W = state['W']
    R0_list = state['R0_list']
    Sh = state['Sh_list']
    Se = state['Se_list']
    k = state['kpoints']
    Uv = state['U_val']
    Uc = state['U_cond']

    Nk = k.shape[0]
    NRh = Sh.shape[0]
    NRe = Se.shape[0]
    Nv = Uv.shape[2]
    Nc = Uc.shape[2]

    # Build transformer for kpq mapping and Wtilde
    ft = BSEWannierFT(kpoints=k, Q=q_vec, U_val=Uv, U_cond=Uc, Sh_list=Sh, Se_list=Se, decimals=state.get('decimals', 10))
    iq_candidates = np.where(ft.kpq[ik] == ikp)[0]
    assert iq_candidates.size == 1, "Ambiguous or missing iq for given (ik,ikp) [alt]"
    iq = int(iq_candidates[0])

    Wtilde = ft.W_to_wtilde(W, R0_list)
    Wq = Wtilde[:, :, iq, :, :]

    # Alternative phases: exp(+i2π[k·Sh + (k+Q)·Se])
    phases = np.exp(+2j * np.pi * (k[ik] @ Sh.T))[:, None] * np.exp(+2j * np.pi * ((k[ik] + q_vec) @ Se.T))[None, :]
    scale = Nk / float(NRh * NRe)
    W_block = scale * np.einsum('rs,rsnm->nm', phases, Wq, optimize=True)

    # Alternative L without Q-shift in valence
    tens_k = np.einsum('mv,nc->mnvc', Uv[ik], Uc[ik], optimize=True)
    Lk = tens_k.reshape(Uv.shape[1]*Uc.shape[1], Nv*Nc)
    tens_kp = np.einsum('mv,nc->mnvc', Uv[ikp], Uc[ikp], optimize=True)
    Lkp = tens_kp.reshape(Uv.shape[1]*Uc.shape[1], Nv*Nc)

    Ktc = Lk.conj().T @ W_block @ Lkp
    return Ktc.reshape(Nv, Nc, Nv, Nc)


def main():
    # Adjust to your environment if needed
    WORK_PATH = '/mnt/lscratch/users/rreho/LiF'
    NK = 3

    # Lattice, TB model and grids
    lat_k = YamboLatticeDB.from_db_file(folder=f'{WORK_PATH}/no-symm/Optics/8bands/{NK}x{NK}x{NK}/SAVE')
    model = TBMODEL.from_wannier_files(
        hr_file=f'{WORK_PATH}/nscf-wannier-{NK}x{NK}x{NK}/LiF_hr.dat',
        wsvec_file=f'{WORK_PATH}/nscf-wannier-{NK}x{NK}x{NK}/LiF_wsvec.dat',
        win_file=f'{WORK_PATH}/nscf-wannier-{NK}x{NK}x{NK}/LiF.win'
    )
    nnkp_kgrid = NNKP_Grids(f'{WORK_PATH}/nscf-wannier-{NK}x{NK}x{NK}//LiF')
    model.set_mpgrid(nnkp_kgrid)
    fermie = 1.0
    hrk = HR_reader(f'{WORK_PATH}/nscf-wannier-{NK}x{NK}x{NK}/LiF')
    model.solve_ham_from_hr(lat_k, hrk, fermie=fermie)

    # Build W from Gamma
    start_iq = 1
    folder_bse = f'{WORK_PATH}/no-symm/Optics/8bands/{NK}x{NK}x{NK}/bse'
    bsk_db = YamboBSEKernelDB.from_db_file(lat_k, Qpt=start_iq, folder=folder_bse)
    exc_db = YamboExcitonDB.from_db_file(lat_k, folder=folder_bse, filename=f'ndb.BS_diago_Q{start_iq}')
    state = build_W_from_yambo_kernel(lat_k, model, bsk_db, exc_db, decimals=6)

    # Print Coulomb cutoff at Γ if available
    try:
        print(f"[cutoff] Gamma q_cutoff={exc_db.q_cutoff}")
    except Exception:
        pass

    # Γ control
    gamma_control(lat_k, state, folder_bse)

    # Optional: build exact on-grid Wq cache from Yambo for all q
    state['Wq_cache'] = build_yambo_wtilde_cache(state, lat_k, folder_bse)

    # Choose a target Q (Python index) != Gamma
    iqwanted = 4
    q_vec = lat_k.red_kpoints[iqwanted]

    # Load reference kernel at this Q
    exc_db_q = YamboExcitonDB.from_db_file(lat_k, folder=folder_bse, filename=f'ndb.BS_diago_Q{iqwanted+1}')
    # Print Coulomb cutoff at target Q if available
    try:
        print(f"[cutoff] Q{iqwanted+1} q_cutoff={exc_db_q.q_cutoff}")
    except Exception:
        pass
    kernel_db_q = YamboBSEKernelDB.from_db_file(lat_k, Qpt=iqwanted+1, folder=folder_bse)
    K6_ref, val_bands_ref, cond_bands_ref = kernel_db_q.as_band_kernel_6d(exc_db_q)

    # Ensure band sets match state (assert; the round-trip build enforces it earlier)
    assert np.all(val_bands_ref == state['val_bands']) and np.all(cond_bands_ref == state['cond_bands'])

    # Pick a single pair (ik,ikp) via k' = k + q mapping using the same table as the transformer
    ft = BSEWannierFT(kpoints=state['kpoints'], Q=q_vec, U_val=state['U_val'], U_cond=state['U_cond'], Sh_list=state['Sh_list'], Se_list=state['Se_list'], decimals=state.get('decimals', 10))
    ik = 0
    iq_match = iqwanted  # exact kpq selection for the chosen q
    ikp = int(ft.kpq[ik, iq_match])

    # Compute explicit block
    K_block_explicit = compute_single_block_explicit(state, q_vec, ik, ikp)

    # Compute inverse via transformer using the same Wtilde for ALL q, then slice
    ft_full = BSEWannierFT(kpoints=state['kpoints'], Q=q_vec, U_val=state['U_val'], U_cond=state['U_cond'], Sh_list=state['Sh_list'], Se_list=state['Se_list'], decimals=state.get('decimals', 10))
    Wtilde_allq = BSEWannierFT.W_to_wtilde_at_q_arbitrary(state['W'], state['R0_list'], q_list=state['kpoints'])
    K6_from_wtilde = ft_full.wtilde_to_bloch(Wtilde_allq)
    K_block_ft = K6_from_wtilde[ik, ikp]

    # Reference block from Yambo
    K_ref_block = K6_ref[ik, ikp]

    # Diffs: explicit vs transformer; transformer vs Yambo
    max_e_ft, mean_e_ft, rel_e_ft = stats(K_block_explicit, K_block_ft)
    max_ft_ref, mean_ft_ref, rel_ft_ref = stats(K_block_ft, K_ref_block)

    # Alternative convention test vs Yambo
    K_block_alt = compute_single_block_explicit_alt(state, q_vec, ik, ikp)
    max_alt_ref, mean_alt_ref, rel_alt_ref = stats(K_block_alt, K_ref_block)

    print(f"ik={ik}, ikp={ikp}, iq_match={iq_match}")
    print("Band sets (state vs ref):")
    print("  val:", state['val_bands'], "|", val_bands_ref)
    print("  cond:", state['cond_bands'], "|", cond_bands_ref)
    print("Shapes:", K_block_explicit.shape, K_block_ft.shape, K_ref_block.shape)
    print("Explicit vs FT  -> Max |diff|:", max_e_ft, "Mean|diff|:", mean_e_ft, "Rel max:", rel_e_ft)
    print("FT vs Yambo     -> Max |diff|:", max_ft_ref, "Mean|diff|:", mean_ft_ref, "Rel max:", rel_ft_ref)
    print("ALT vs Yambo    -> Max |diff|:", max_alt_ref, "Mean|diff|:", mean_alt_ref, "Rel max:", rel_alt_ref)

    # Report Yambo mapping for this block
    report_yambo_block_mapping(exc_db_q, ik, ikp, max_print=8)

    # Direct block from triangular kernel using exact transition indices
    K_ref_block_tri = block_from_triangular(kernel_db_q, exc_db_q, ik, ikp, val_bands_ref, cond_bands_ref)
    tri_vs_ref = stats(K_ref_block_tri, K_ref_block)
    print("Triangular-sliced vs as_band_kernel_6d -> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % tri_vs_ref)

    # q-sign flip diagnostic
    K_block_negq = compute_block_via_neg_q(state, q_vec, ik, ikp)
    negq_vs_ref = stats(K_block_negq, K_ref_block)
    print("Neg-q Hermitian vs Yambo -> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % negq_vs_ref)

    # Axis permutation diagnostics
    perms = axis_permutation_diffs(K_block_ft, K_ref_block)
    best = min(perms.items(), key=lambda kv: kv[1][0])  # by max_abs
    print("Axis-permutation best match (using FT block vs Yambo):", best[0], "-> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % best[1])

    # Optional: evaluate K via offgrid API (should match FT when q is on-grid)
    K_per_k_off, kpq_used = evaluate_offgrid_K_from_state(state, q_vec, model)
    K_block_off = K_per_k_off[ik]
    off_vs_ref = stats(K_block_off, K_ref_block)
    print("Offgrid-eval (on-grid path) vs Yambo -> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % off_vs_ref)

    # Q from DB vs selected q (check for any mismatch)
    try:
        Qcar = np.asarray(exc_db_q.car_qpoint, float)
        Qred_db = car_red(Qcar, lat_k.rlat)
        # wrap difference to [-0.5,0.5) before norm
        dq = np.linalg.norm(((Qred_db - q_vec + 0.5) % 1.0) - 0.5)
        print(f"Q from DB (red): {Qred_db}, selected q_vec: {q_vec}, |Δ| (wrapped) = {dq:.3e}")
    except Exception as e:
        print("[warn] Could not compare DB Q to selected q:", e)

    # Compare Wtilde derived from Yambo K vs Wtilde evaluated from our W at the same q
    ft_ref = BSEWannierFT(kpoints=state['kpoints'], Q=q_vec, U_val=state['U_val'], U_cond=state['U_cond'],
                          Sh_list=state['Sh_list'], Se_list=state['Se_list'], decimals=state.get('decimals', 10))
    Wtilde_ref = ft_ref.bloch_to_wtilde(K6_ref)
    Wtilde_state = ft_ref.W_to_wtilde(state['W'], state['R0_list'])
    Wq_ref = Wtilde_ref[:, :, iq_match, :, :]
    Wq_state = Wtilde_state[:, :, iq_match, :, :]
    wq_max, wq_mean, wq_rel = stats(Wq_state, Wq_ref)
    ma, sa, mph, sph = phase_amplitude_drift(Wq_state, Wq_ref)
    print(f"Wtilde slab (state vs ref) @iq={iq_match}: Max |diff|={wq_max:.3e}, Mean={wq_mean:.3e}, Rel={wq_rel:.3e}; |ratio| mean±std={ma:.3e}±{sa:.3e}, phase mean±std={mph:.3e}±{sph:.3e}")

    # Reconstruct K block from Wtilde_ref using our inverse and compare to Yambo block
    K_block_from_Wq_ref = compute_single_block_from_Wq(state, q_vec, ik, ikp, Wq_ref)
    refblock_vs_y = stats(K_block_from_Wq_ref, K_ref_block)
    print("K block from Wtilde_ref vs Yambo -> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % refblock_vs_y)

    # Sanity: reconstruct K block from Wtilde_state and compare to FT block
    K_block_from_Wq_state = compute_single_block_from_Wq(state, q_vec, ik, ikp, Wq_state)
    stateblock_vs_ft = stats(K_block_from_Wq_state, K_block_ft)
    print("K block from Wtilde_state vs FT -> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % stateblock_vs_ft)

    # --- Test 2: Build W at this Q from Yambo K(Q) and compare ---
    W_Q, K_back_Q, R0_Q = kernel_realspace_roundtrip(
        K6_ref,
        kpoints=state['kpoints'],
        Q=q_vec,
        U_val=state['U_val'],
        U_cond=state['U_cond'],
        Sh_list=state['Sh_list'],
        Se_list=state['Se_list'],
        R0_list=state['R0_list'],
        decimals=state.get('decimals', 10),
    )
    # Check back-projection at (ik,ikp)
    backblock_vs_y = stats(K_back_Q[ik, ikp], K_ref_block)
    print("[Test2] K_back_Q block vs Yambo -> Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % backblock_vs_y)

    # Build Wtilde slab from W_Q at the same iq and compare to Wq_ref
    Wtilde_from_W_Q_all = BSEWannierFT.W_to_wtilde_at_q_arbitrary(W_Q, R0_Q, q_list=state['kpoints'])
    Wq_from_W_Q = Wtilde_from_W_Q_all[:, :, iq_match, :, :]
    wq_fromWQ_vs_ref = stats(Wq_from_W_Q, Wq_ref)
    ma2, sa2, mph2, sph2 = phase_amplitude_drift(Wq_from_W_Q, Wq_ref)
    print(f"[Test2] Wq_from_W_Q vs Wq_ref: Max |diff|={wq_fromWQ_vs_ref[0]:.3e}, Mean={wq_fromWQ_vs_ref[1]:.3e}, Rel={wq_fromWQ_vs_ref[2]:.3e}; |ratio| mean±std={ma2:.3e}±{sa2:.3e}, phase mean±std={mph2:.3e}±{sph2:.3e}")

    # Compare real-space kernels W_Q vs state['W'] to detect Q-dependence at real-space level
    W_vs_state = stats(W_Q, state['W'])
    print("[Test2] W(Q) vs W(Gamma-built): Max |diff|=%.3e, Mean=%.3e, Rel=%.3e" % W_vs_state)


if __name__ == '__main__':
    main()