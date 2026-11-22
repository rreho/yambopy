# Copyright (c) 2026, University of Luxembourg
# All rights reserved.
#
# This file is part of yambopy
#
# Author: RR
#
"""
Exciton total crystal angular momentum (TCAM) and real-space L_z density.

Theory
------
If the little group of the exciton momentum Q contains an n-fold rotation
C_n about the axis n^, its representation matrix in the exciton basis,

    D_{mn}(C_n) = e^{i Q.tau} sum_{kcv} A^{m*}_{kcv} [C_n A^n]_{kcv},
    [g A]_{Rk,cv} = sum_{c'v'} D^c_{cc'}(g,k) A_{k,c'v'} D^{v*}_{vv'}(g,k-Q),

(the matrix built by exciton_irreps.compute_exc_rep / rotate_exc_wf) is unitary
and can be written as

    D(C_n) = exp(-i 2 pi J_n / n),

which defines the Hermitian TCAM operator J_n. [J_n, H_BSE] = 0, so each
exciton carries j = 0..n-1 (mod n), here mapped to the window (-n/2, n/2].
Eigenvalues of D restricted to a (complete) degenerate multiplet give

    lambda = exp(-2 pi i j / n)   ->   j = -n arg(lambda) / (2 pi)   (mod n).

Selection rule: j_S' = j_S + j_lambda (mod n). The in-plane photon e_+ =
(x + i y)/sqrt(2) has j = +1, so a state whose emission dipole
D_S = sum A^S_kcv <v|r|c> is along e_+ (circular_basis.E_PLUS) has j = +1.
With spinor Dmats j is the total (orbital + spin) quantum number.

Phonons: for g = {R|t} with Rq = q + G, atom kappa -> R tau_k + t = tau_k' + L_k
(L_k a lattice vector). With u_k(R_l) = e_k e^{iq.R_l} (QE/LetzElPhC 'cell'
gauge) the actively rotated mode is

    [g e]_k' = e^{-i q.L_k} R e_k,       D_{nu mu}(g) = <e_nu| g e_mu>.

In the 'atom' gauge u_k(R_l) = e~_k e^{iq.(R_l+tau_k)} one uses e = e~ e^{iq.tau}
(same D; the two differ only if g permutes atoms, otherwise the per-atom phase
commutes with g). lambda_nu = exp(-2 pi i j_nu / n) defines the phonon pseudo-angular
momentum; it depends on the rotation centre (t), which must be the same one
used for the excitons. dV_nu is linear in e_nu, so U dV_nu U^+ = sum D dV and

    G^nu_{S'S} = <S'|dV_nu|S> = lam*_{S'} lam_nu lam_S G^nu_{S'S}
    ->  G^nu_{S'S} = 0  unless  j_S' = j_S + j_nu (mod n).

Real space: Psi_S(r_e; r_h) = sum_kcv A^S_kcv psi_ck(r_e) psi*_vk(r_h) is
defined only up to a global phase, so Re/Im cubes are gauge dependent. The
gauge-invariant hole-centred angular-momentum density (atomic units, hbar=1)

    j(r)    = Im[ Psi^* grad Psi ](r)
    l_z(r)  = (x - x_h) j_y(r) - (y - y_h) j_x(r) = Re[Psi^* L_z Psi],
    <L_z>   = int l_z / int |Psi|^2   (units of hbar, L_z = -i d/dphi)

has one sign per chirality. For Psi ~ (x + i y) e^{-r^2}, <L_z> = +1.
Only the C_n eigenvalue of Psi about an axis through r_h is protected (mod n);
<L_z> itself is not quantised in a crystal.
"""
import numpy as np


def tcam_from_rotation(D_Cn, n, unit_tol=1e-2):
    """
    TCAM quantum numbers from the C_n representation matrix of a set of states.

    Parameters
    ----------
    D_Cn : (N, N) complex
        D_mn = <A_m | C_n | A_n>; must span complete degenerate multiplets.
    n : int
        Order of the rotation (2 pi / n angle).
    unit_tol : float
        Warn if |lambda| deviates from 1 by more than this (subspace not
        closed under C_n, e.g. a truncated multiplet or broken symmetry).

    Returns
    -------
    j : (N,) float
        j = -n arg(lambda) / (2 pi), mapped to (-n/2, n/2]. Non-integer values
        signal symmetry breaking.
    lam : (N,) complex
        Eigenvalues of D_Cn.
    V : (N, N) complex
        Eigenvectors (columns): |j> = sum_m V[m, j] |A_m>.
    """
    lam, V = np.linalg.eig(np.asarray(D_Cn))
    if np.any(np.abs(np.abs(lam) - 1.0) > unit_tol):
        print("[tcam] Warning: |lambda| != 1 -> subspace not C%d invariant: %s"
              % (n, np.round(np.abs(lam), 4)))
    j = -n * np.angle(lam) / (2.0 * np.pi)          # in [-n/2, n/2]
    j = np.where(j <= -n / 2.0 + 1e-6, j + n, j)    # -> (-n/2, n/2]
    order = np.argsort(-j)
    return j[order], lam[order], V[:, order]


def find_rotation(rotations, n, axis=(0, 0, 1), tol=1e-5):
    """
    Index of the proper, counter-clockwise rotation by 2 pi/n about `axis`
    in a list of Cartesian rotation matrices (e.g. Crystal_Symmetries.rotations).
    Returns None if not present.
    """
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    th = 2.0 * np.pi / n
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R0 = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K   # Rodrigues
    for i, R in enumerate(rotations):
        if np.allclose(R, R0, atol=tol):
            return i
    return None


def exciton_rotation_rep(Akcv, R_cart, wfdb, dmats, Qpt_crys=None,
                         tau_cart=None, Akcv_left=None):
    """
    Representation matrix D_mn = e^{iQ.tau} <A_m| g |A_n> of one symmetry
    g = {R|tau} on the exciton states Akcv (same as compute_exc_rep).

    Parameters
    ----------
    Akcv : (N, 1|2, nspin, nk, nc, nv) from YamboExcitonDB.get_Akcv()
    R_cart : (3, 3) Cartesian rotation
    wfdb : YamboWFDB (provides lattice, kBZ, ktree)
    dmats : (nk, nspin, nb, nb) electronic rep. matrices of g,
            e.g. wfdb.Dmat(symm_mat=R[None], frac_vec=tau[None], time_rev=False)[0]
    Qpt_crys : exciton momentum in crystal units (default Gamma)
    Akcv_left : left eigenvectors for non-TDA (default: Akcv)
    """
    from yambopy.bse.rotate_excitonwf import rotate_exc_wf
    lat = wfdb.ydb.lat
    Q = np.zeros(3) if Qpt_crys is None else np.asarray(Qpt_crys, float)
    tau = np.zeros(3) if tau_cart is None else np.asarray(tau_cart, float)
    R_red = lat @ np.asarray(R_cart) @ np.linalg.inv(lat)
    rot = rotate_exc_wf(Akcv, R_red, wfdb.kBZ, Q, dmats, False, ktree=wfdb.ktree)
    left = Akcv if Akcv_left is None else Akcv_left
    phase = np.exp(2j * np.pi * (np.linalg.inv(lat) @ Q) @ tau)
    return phase * np.einsum('m...,n...->mn', left.conj(), rot, optimize=True)


def tcam_basis(D_Cn, n, unit_tol=1e-2):
    """
    tcam_from_rotation with an orthonormal eigenbasis (QR within equal j):
    returns j (N,), U (N, N) unitary, |j_a> = sum_m U[m, a] |m>.
    """
    j, _, V = tcam_from_rotation(D_Cn, n, unit_tol)
    for jj in np.unique(np.rint(j)):
        s = np.rint(j) == jj
        V[:, s] = np.linalg.qr(V[:, s])[0]
    return j, V


def phonon_tcam(ph_eigvecs, ph_energies, atom_pos, lat, R_cart, n, q_crys=None,
                t_cart=None, gauge='cell', degen_tol=1e-4):
    """
    Phonon pseudo-angular momentum j_nu for the C_n rotation {R_cart|t_cart}.

    Parameters
    ----------
    ph_eigvecs : (nm, nat, 3) complex, Cartesian (eigenvectors or e/sqrt(M))
    ph_energies : (nm,) any unit, used only for degenerate blocks (degen_tol)
    atom_pos : (nat, 3) Cartesian positions, same units as lat
    lat : (3, 3) lattice vectors as ROWS (YamboLatticeDB.lat)
    R_cart : (3, 3) Cartesian rotation; n its order
    q_crys : phonon momentum in crystal units (default Gamma)
    t_cart : translation of the operation x -> Rx + t (default 0)
    gauge : 'cell' (e^{iq.R}, QE/LetzElPhC) or 'atom' (e^{iq.(R+tau)})

    Returns
    -------
    j : (nm,) j_nu in (-n/2, n/2]
    W : (nm, nm) unitary within degenerate blocks: mode a = sum_mu W[mu, a] e_mu
    """
    e = np.asarray(ph_eigvecs, complex)
    tau = np.asarray(atom_pos, float)
    lat = np.asarray(lat, float)
    q = np.zeros(3) if q_crys is None else np.asarray(q_crys, float)
    t = np.zeros(3) if t_cart is None else np.asarray(t_cart, float)
    q_cart = 2 * np.pi * np.linalg.inv(lat) @ q
    dG = lat @ (R_cart - np.eye(3)) @ q_cart / (2 * np.pi)         # Rq - q, crystal
    assert np.allclose(dG, np.rint(dG), atol=1e-5), "R q != q + G"
    if gauge == 'atom': e = e * np.exp(1j * tau @ q_cart)[None, :, None]
    gE = np.zeros_like(e)
    for k in range(len(tau)):
        new = R_cart @ tau[k] + t
        fr = np.linalg.solve(lat.T, (new[None, :] - tau).T).T     # crystal units
        kp = np.where(np.all(np.abs(fr - np.rint(fr)) < 1e-4, axis=1))[0]
        assert len(kp) == 1, "atom %d not mapped onto an atom" % k
        L = new - tau[kp[0]]
        gE[:, kp[0]] = np.exp(-1j * q_cart @ L) * e[:, k] @ R_cart.T
    nm = e.shape[0]
    E, gE = e.reshape(nm, -1).T, gE.reshape(nm, -1).T
    D = np.linalg.lstsq(E, gE, rcond=None)[0]          # g e_mu = sum_nu e_nu D_{nu mu}
    j, W = np.zeros(nm), np.zeros((nm, nm), complex)
    w = np.asarray(ph_energies, float)
    for b in np.split(np.arange(nm), np.where(np.diff(w) > degen_tol)[0] + 1):
        j[b], W[np.ix_(b, b)] = tcam_basis(D[np.ix_(b, b)], n)
    return j, W


def check_tcam_conservation(G, j_out, j_in, j_ph, n, tol=0.1):
    """
    Diagnostic of the TCAM selection rule j_out = j_in + j_ph (mod n).

    G : (nmodes, n_in, n_out), Ex-ph.npy order, G[nu, i, o] = <o|dV_nu|i>,
        already in TCAM bases: G_j = einsum('ma,nb,oc,mno->abc', W, U_in,
        U_out.conj(), G) with W from phonon_tcam, U from tcam_basis.
    tol : |j_out - j_in - j_ph| mod n below tol counts as allowed.

    Returns the violating |G|^2 weight fraction, max violating |G| and its
    index (nu, i, o).
    """
    G = np.asarray(G)
    dj = (np.asarray(j_out)[None, None, :] - np.asarray(j_in)[None, :, None]
          - np.asarray(j_ph)[:, None, None])
    dj = np.abs((dj + n / 2.0) % n - n / 2.0)
    G2 = np.abs(G)**2
    bad = np.broadcast_to(dj > tol, G.shape)
    frac = G2[bad].sum() / max(G2.sum(), 1e-300)
    if frac == 0: return 0.0, 0.0, None
    idx = np.unravel_index(np.argmax(np.where(bad, G2, -1)), G.shape)
    return frac, np.abs(G[idx]), idx


def lz_density(psi, lat_vec, center, sigma=None):
    """
    Gauge-invariant hole-centred angular-momentum density l_z(r).

    Parameters
    ----------
    psi : (..., Nx, Ny, Nz) complex
        Wavefunction on the uniform grid spanning lat_vec (origin at 0).
        Leading axes (spinor channels, as returned by ex_wf2Real per state)
        are summed in l_z and |psi|^2.
    lat_vec : (3, 3) cell vectors as COLUMNS (bohr), e.g. ex_wf2Real output.
    center : (3,) Cartesian rotation centre (bohr), e.g. the hole position
        (last atom returned by ex_wf2Real). Axis is z.
    sigma : float or None
        Optional Gaussian averaging length (bohr) applied to l_z.

    Returns
    -------
    lz : (Nx, Ny, Nz) real
    Lz_mean : float, <L_z> = sum l_z / sum |psi|^2 in units of hbar.
    """
    psi = np.asarray(psi, dtype=complex)
    shape = psi.shape[-3:]
    psi = psi.reshape((-1,) + shape)
    A = np.asarray(lat_vec, float)
    B = 2.0 * np.pi * np.linalg.inv(A).T                 # columns b_i
    g = np.meshgrid(*[np.fft.fftfreq(N) * N for N in shape], indexing='ij')
    KX = sum(B[0, i] * g[i] for i in range(3))
    KY = sum(B[1, i] * g[i] for i in range(3))
    # minimum-image Cartesian displacement from the centre
    fr = np.meshgrid(*[np.arange(N) / N for N in shape], indexing='ij')
    fd = np.stack(fr) - (np.linalg.inv(A) @ np.asarray(center, float))[:, None, None, None]
    fd -= np.rint(fd)
    dR = np.einsum('ab,bxyz->axyz', A, fd)
    lz = np.zeros(shape)
    for p in psi:
        F = np.fft.fftn(p)
        jx = np.imag(np.conj(p) * np.fft.ifftn(1j * KX * F))
        jy = np.imag(np.conj(p) * np.fft.ifftn(1j * KY * F))
        lz += dR[0] * jy - dR[1] * jx
    Lz_mean = float(lz.sum() / (np.abs(psi)**2).sum())
    if sigma is not None:
        KZ = sum(B[2, i] * g[i] for i in range(3))
        G = np.exp(-0.5 * (KX**2 + KY**2 + KZ**2) * sigma**2)
        lz = np.real(np.fft.ifftn(np.fft.fftn(lz) * G))
    return lz, Lz_mean


def write_lz_cube(filename, psi, lat_vec, atom_pos, atom_nums, center,
                  sigma=None, header=''):
    """
    Compute l_z (lz_density) and write it, normalised to max|l_z| = 1, to a
    Gaussian cube via io.cubetools.write_cube. Returns <L_z> (hbar).
    """
    from yambopy.io.cubetools import write_cube
    lz, Lz_mean = lz_density(psi, lat_vec, center, sigma=sigma)
    write_cube(filename, lz / max(np.abs(lz).max(), 1e-300), lat_vec,
               atom_pos, atom_nums,
               header=header or 'hole-centred l_z = Im[Psi*(x dy - y dx)Psi], <Lz>=%.4f'
               % Lz_mean)
    return Lz_mean
