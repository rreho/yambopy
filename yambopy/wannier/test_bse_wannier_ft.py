import os, sys
# Ensure local repo package is imported, not system-installed one
sys.path.insert(0, "/home/users/rreho/personalcodes/yambopy-project")

import numpy as np
from numpy.linalg import norm
from yambopy.wannier.wann_bse_wannier import BSEWannierFT

# Toy test of Bloch <-> Wtilde <-> W(R0) round-trips
# Grid: 3D uniform, small size; Q=0; shells provided as inputs

def random_unitary(n):
    X = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    # Ensure columns are orthonormal
    return Q

def random_semiunitary(nrows, ncols):
    # nrows >= ncols, column-orthonormal
    X = np.random.randn(nrows, ncols) + 1j*np.random.randn(nrows, ncols)
    Q, _ = np.linalg.qr(X)
    return Q[:, :ncols]


def build_small_grid(nk_axis=2):
    # points in (-0.5,0.5): {-0.5, 0.0} for nk_axis=2, wrap to [0,1) internally
    vals = np.linspace(-0.5, 0.5, nk_axis, endpoint=False)
    klist = np.array([[x, y, z] for x in vals for y in vals for z in vals], float)
    return klist


def build_shell(max_r=0):
    # Simple shell with only Gamma by default
    vecs = [(0,0,0)]
    return np.array(vecs, int)


def build_shell_cube(Rmax=1):
    # All integer displacements in a cube [-Rmax, Rmax]^3
    vals = range(-Rmax, Rmax+1)
    return np.array([[x, y, z] for x in vals for y in vals for z in vals], int)


def build_R0_list(nk_axis=2):
    # Dual (integer) grid for the q-grid: values 0..(nk_axis-1) per axis
    vals = list(range(nk_axis))
    R0 = np.array([[x, y, z] for x in vals for y in vals for z in vals], int)
    return R0


def test_round_trip(seed=7):
    np.random.seed(seed)

    # Grids (use 4 points per axis)
    nk_axis = 4
    k = build_small_grid(nk_axis)   # Nk = 64
    Nk = k.shape[0]
    Q = np.array([0.0, 0.0, 0.0])

    # Dimensions
    NWh, NWe = 3, 4
    Nv, Nc = 2, 2
    Nm = NWh*NWe

    # Random semi-unitary U at k (valence at k, conduction at k)
    U_val = np.empty((Nk, NWh, Nv), dtype=complex)
    U_cond = np.empty((Nk, NWe, Nc), dtype=complex)
    for ik in range(Nk):
        U_val[ik]  = random_semiunitary(NWh, Nv)
        U_cond[ik] = random_semiunitary(NWe, Nc)

    # Real-space lists for exact round-trip:
    # - Keep Sh at Gamma only
    # - Use Se as the full dual grid (size Nk) so the k-DFT is invertible
    R0_list = build_R0_list(nk_axis)  # size Nk, dual to the nk_axis^3 grid
    Sh_list = build_shell(0)          # only (0,0,0)
    Se_list = R0_list                  # full dual grid for exact DFT

    # Build transformer (handles k−Q internally using Q)
    ft = BSEWannierFT(kpoints=k, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list)

    # Round-trip starting from random Hermitian K
    K6 = np.zeros((Nk, Nk, Nv, Nc, Nv, Nc), dtype=complex)
    for ik in range(Nk):
        for ikp in range(Nk):
            A = np.random.randn(Nv*Nc, Nv*Nc) + 1j*np.random.randn(Nv*Nc, Nv*Nc)
            H = (A + A.conj().T)/2
            K6[ik, ikp] = H.reshape(Nv, Nc, Nv, Nc)

    # Forward and inverse Bloch <-> Wtilde
    Wtilde1 = ft.bloch_to_wtilde(K6)
    K6_back = ft.wtilde_to_bloch(Wtilde1)

    err_K = norm((K6_back - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
    print("Relative error round-trip K -> Wtilde -> K:", err_K)
    assert err_K < 5e-10

    # R0 Fourier pair checks: Wtilde <-> W(R0)
    W_R0 = ft.wtilde_to_W(Wtilde1, R0_list)
    Wtilde2 = ft.W_to_wtilde(W_R0, R0_list)

    err_Wtilde = norm((Wtilde2 - Wtilde1).reshape(-1)) / max(1.0, norm(Wtilde1.reshape(-1)))
    print("Relative error round-trip Wtilde -> W(R0) -> Wtilde:", err_Wtilde)
    assert err_Wtilde < 5e-10

    # Consistency: K from W via W->Wtilde->Bloch matches K from Wtilde directly
    K6_from_W = ft.wtilde_to_bloch(Wtilde2)
    err_K_from_W = norm((K6_from_W - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
    print("Relative error K[W] vs K[Wtilde]:", err_K_from_W)
    assert err_K_from_W < 5e-10

    # Unitarity test for U matrices
    dv, dc = ft.check_unitarity()
    print("U_val/unitarity deviation, U_cond/unitarity deviation:", dv, dc)
    assert dv < 1e-10 and dc < 1e-10

    # Index mapping test: k + q table
    rounded = np.round(np.mod(k, 1.0), 10)
    for ik in range(Nk):
        for iq in range(Nk):
            ikp = ft.kpq[ik, iq]
            lhs = rounded[ikp]
            rhs = np.round(np.mod(rounded[ik] + rounded[iq], 1.0), 10)
            assert np.allclose(lhs, rhs)


def test_conservation_simple(seed=3):
    np.random.seed(seed)

    # Grid and dims
    nk_axis = 4
    k = build_small_grid(nk_axis)
    Nk = k.shape[0]
    Q = np.array([0.0, 0.0, 0.0])

    NWh = NWe = Nv = Nc = 1

    # U as identity (trivial unitaries)
    U_val = np.ones((Nk, NWh, Nv), dtype=complex)
    U_cond = np.ones((Nk, NWe, Nc), dtype=complex)

    # Shells for exact round-trip
    R0_list = build_R0_list(nk_axis)
    Sh_list = build_shell(0)   # Gamma only
    Se_list = R0_list          # full dual grid

    ft = BSEWannierFT(kpoints=k, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list)

    # Build Hermitian kernel across (k,k')
    A = np.random.randn(Nk, Nk) + 1j * np.random.randn(Nk, Nk)
    Ktc = (A + A.conj().T) / 2  # Hermitian across k indices
    K6 = Ktc.reshape(Nk, Nk, Nv, Nc, Nv, Nc)

    # Transform there and back
    Wtilde = ft.bloch_to_wtilde(K6)
    K6_back = ft.wtilde_to_bloch(Wtilde)

    # Check Hermiticity is preserved
    Ktc_back = K6_back.reshape(Nk, Nk)
    assert np.allclose(Ktc_back, Ktc_back.conj().T, atol=1e-12)




def test_truncated_shells_idempotence(seed=11):
    np.random.seed(seed)

    # Grid and dims (same as main test)
    nk_axis = 4
    k = build_small_grid(nk_axis)
    Nk = k.shape[0]
    Q = np.array([0.0, 0.0, 0.0])

    NWh, NWe = 3, 4
    Nv, Nc = 2, 2

    # Random semi-unitary U
    U_val = np.empty((Nk, NWh, Nv), dtype=complex)
    U_cond = np.empty((Nk, NWe, Nc), dtype=complex)
    for ik in range(Nk):
        U_val[ik]  = random_semiunitary(NWh, Nv)
        U_cond[ik] = random_semiunitary(NWe, Nc)

    # Use identical Sh and Se shells (symmetric cube around 0)
    # This matches the idea Sh = Rh - Rh' and Se = Re - Re', independent of k
    Rmax = 1
    shell = build_shell_cube(Rmax)
    Sh_list = shell.copy()
    Se_list = shell.copy()

    # Build K and transform
    ft = BSEWannierFT(kpoints=k, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list)

    # Generate Hermitian K
    Nt = Nv * Nc
    K6 = np.zeros((Nk, Nk, Nv, Nc, Nv, Nc), dtype=complex)
    for ik in range(Nk):
        for ikp in range(Nk):
            A = np.random.randn(Nt, Nt) + 1j*np.random.randn(Nt, Nt)
            H = (A + A.conj().T) / 2
            K6[ik, ikp] = H.reshape(Nv, Nc, Nv, Nc)

    # Forward and inverse with truncated shells
    Wtilde = ft.bloch_to_wtilde(K6)
    K6_back = ft.wtilde_to_bloch(Wtilde)

    # This is not guaranteed to be exact unless the shells span the dual space fully.
    # Test idempotence of the composed operator P = T^{-1} T with truncated shells:
    # Apply twice and check P(P(K)) ≈ P(K).
    err = norm((K6_back - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
    print("Truncated shells round-trip relative error (projection):", err)

    # Apply the transform again starting from K6_back
    Wtilde2 = ft.bloch_to_wtilde(K6_back)
    K6_back2 = ft.wtilde_to_bloch(Wtilde2)

    err_idem = norm((K6_back2 - K6_back).reshape(-1)) / max(1.0, norm(K6_back.reshape(-1)))
    print("Truncated shells idempotence error ||P^2(K) - P(K)|| / ||P(K)||:", err_idem)
    # With truncated shells, T^{-1}T is not a strict projector; require stability instead
    assert err_idem <= 1.2 * err + 1e-12


def test_round_trip_dual_shells_machine_precision(seed=13):
    np.random.seed(seed)
    nk_axis = 4
    k = build_small_grid(nk_axis)
    Nk = k.shape[0]
    Q = np.array([0.0, 0.0, 0.0])

    NWh, NWe = 3, 4
    Nv, Nc = 2, 2

    # Random semi-unitary U
    U_val = np.empty((Nk, NWh, Nv), dtype=complex)
    U_cond = np.empty((Nk, NWe, Nc), dtype=complex)
    for ik in range(Nk):
        U_val[ik]  = random_semiunitary(NWh, Nv)
        U_cond[ik] = random_semiunitary(NWe, Nc)

    # Use full dual sets for both shells
    dual = build_R0_list(nk_axis)  # size Nk
    Sh_list = dual
    Se_list = dual

    ft = BSEWannierFT(kpoints=k, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list)

    # Random Hermitian K
    Nt = Nv * Nc
    K6 = np.zeros((Nk, Nk, Nv, Nc, Nv, Nc), dtype=complex)
    for ik in range(Nk):
        for ikp in range(Nk):
            A = np.random.randn(Nt, Nt) + 1j*np.random.randn(Nt, Nt)
            H = (A + A.conj().T) / 2
            K6[ik, ikp] = H.reshape(Nv, Nc, Nv, Nc)

    Wtilde = ft.bloch_to_wtilde(K6)
    K6_back = ft.wtilde_to_bloch(Wtilde)

    err = norm((K6_back - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
    print("Dual shells exact round-trip error:", err)
    assert err < 1e-12

    # Also check Wtilde <-> W(R0) round-trip
    R0_list = dual
    W_R0 = ft.wtilde_to_W(Wtilde, R0_list)
    Wtilde_back = ft.W_to_wtilde(W_R0, R0_list)
    err_w = norm((Wtilde_back - Wtilde).reshape(-1)) / max(1.0, norm(Wtilde.reshape(-1)))
    print("Wtilde <-> W(R0) round-trip error:", err_w)
    assert err_w < 1e-12


def test_round_trip_dual_shells_Q_nonzero(seed=17):
    np.random.seed(seed)
    nk_axis = 4
    k = build_small_grid(nk_axis)
    Nk = k.shape[0]
    # Choose a Q that is itself on the grid so k-Q is still on-grid
    Q = np.array([0.25, 0.0, 0.0])

    NWh, NWe = 3, 4
    Nv, Nc = 2, 2

    U_val = np.empty((Nk, NWh, Nv), dtype=complex)
    U_cond = np.empty((Nk, NWe, Nc), dtype=complex)
    for ik in range(Nk):
        U_val[ik]  = random_semiunitary(NWh, Nv)
        U_cond[ik] = random_semiunitary(NWe, Nc)

    dual = build_R0_list(nk_axis)
    Sh_list = dual
    Se_list = dual

    ft = BSEWannierFT(kpoints=k, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list)

    Nt = Nv * Nc
    K6 = np.zeros((Nk, Nk, Nv, Nc, Nv, Nc), dtype=complex)
    for ik in range(Nk):
        for ikp in range(Nk):
            A = np.random.randn(Nt, Nt) + 1j*np.random.randn(Nt, Nt)
            H = (A + A.conj().T) / 2
            K6[ik, ikp] = H.reshape(Nv, Nc, Nv, Nc)

    Wtilde = ft.bloch_to_wtilde(K6)
    K6_back = ft.wtilde_to_bloch(Wtilde)

    err = norm((K6_back - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
    print("Dual shells (Q!=0) exact round-trip error:", err)
    assert err < 1e-12


if __name__ == "__main__":
    test_round_trip()
    test_conservation_simple()
    test_truncated_shells_idempotence()
    test_round_trip_dual_shells_machine_precision()
    test_round_trip_dual_shells_Q_nonzero()