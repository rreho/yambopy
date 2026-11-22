# Copyright (c) 2026, University of Luxembourg
# All rights reserved.
#
# This file is part of yambopy
#
import unittest
import numpy as np
from yambopy.bse.circular_basis import E_PLUS, E_MINUS
from yambopy.bse.exciton_angular_momentum import (tcam_from_rotation, find_rotation,
                                                  lz_density, phonon_tcam,
                                                  check_tcam_conservation)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class TestTCAM(unittest.TestCase):

    def test_synthetic_c3_rep(self):
        """Random-basis C3 rep with known j = {+1, 0, -1, +1} is recovered."""
        rng = np.random.default_rng(3)
        j_true = np.array([1, 0, -1, 1])
        lam = np.exp(-2j * np.pi * j_true / 3)
        X = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        Q, _ = np.linalg.qr(X)
        D = Q @ np.diag(lam) @ Q.conj().T
        j, lam_out, V = tcam_from_rotation(D, 3)
        np.testing.assert_allclose(j, [1, 1, 0, -1], atol=1e-10)
        np.testing.assert_allclose(np.abs(lam_out), 1, atol=1e-12)
        np.testing.assert_allclose(D @ V, V * lam_out[None, :], atol=1e-10)

    def test_window_and_c6(self):
        """j is mapped to (-n/2, n/2]: j = 3 for C6 stays +3, j = 4 -> -2."""
        j, _, _ = tcam_from_rotation(np.diag(np.exp(-2j * np.pi * np.array([3, 4, 5]) / 6)), 6)
        np.testing.assert_allclose(sorted(j), [-2, -1, 3], atol=1e-10)

    def test_sigma_plus_is_j_plus_one(self):
        """A vector field c.r transforms with D = R (active rotation); e_+ ~ x+iy -> j=+1."""
        R = _rot_z(2 * np.pi / 3)
        j, lam, V = tcam_from_rotation(R[:2, :2], 3)
        for vec, jj in ((E_PLUS[:2], 1), (E_MINUS[:2], -1)):
            k = np.argmin(np.abs(j - jj))
            self.assertGreater(abs(np.vdot(V[:, k], vec)) / np.linalg.norm(vec), 1 - 1e-12)

    def test_find_rotation(self):
        rots = [np.eye(3), _rot_z(-2 * np.pi / 3), _rot_z(2 * np.pi / 3)]
        self.assertEqual(find_rotation(rots, 3), 2)
        self.assertIsNone(find_rotation(rots, 6))


class TestLzDensity(unittest.TestCase):

    def setUp(self):
        L, N = 12.0, 48
        self.lat = np.diag([L, L, L])
        x = np.arange(N) * L / N
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        self.c = np.array([L / 2, L / 2, L / 2])
        self.X, self.Y = X - self.c[0], Y - self.c[1]
        self.env = np.exp(-(self.X**2 + self.Y**2 + (Z - self.c[2])**2))

    def test_plus_minus_one(self):
        for sgn in (+1, -1):
            psi = (self.X + 1j * sgn * self.Y) * self.env
            lz, Lz = lz_density(psi, self.lat, self.c)
            self.assertAlmostEqual(Lz, sgn, places=6)
            # one sign everywhere it matters
            big = np.abs(lz) > 1e-3 * np.abs(lz).max()
            self.assertTrue(np.all(np.sign(lz[big]) == sgn))

    def test_gauge_invariance_and_spinor_sum(self):
        psi = (self.X + 1j * self.Y) * self.env
        lz0, _ = lz_density(psi, self.lat, self.c)
        lz1, _ = lz_density(np.exp(0.7j) * psi, self.lat, self.c)
        np.testing.assert_allclose(lz1, lz0, atol=1e-12)
        # two channels with opposite chirality and equal weight -> <Lz> = 0
        both = np.stack([psi, (self.X - 1j * self.Y) * self.env])
        _, Lz = lz_density(both, self.lat, self.c)
        self.assertAlmostEqual(Lz, 0.0, places=6)
        # s-like function carries no angular momentum
        _, Lz = lz_density(self.env + 0j, self.lat, self.c)
        self.assertAlmostEqual(Lz, 0.0, places=8)


class TestPhononTCAM(unittest.TestCase):
    """Toy honeycomb (a = 1, A at 0, B at (a1+a2)/3), C3 about A.
    R tau_B = tau_B - a1, so at K = (1/3,-1/3): [C3 e]_B = e^{iK.a1} R e_B = e^{2pi i/3} R e_B.
    R e_+- = e^{-+2pi i/3} e_+-  ->  A+ : +1, A- : -1, Az : 0, B+ : 0, B- : +1, Bz : -1."""

    def setUp(self):
        self.lat = np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 10.0]])
        self.tau = np.array([[0, 0, 0], (self.lat[0] + self.lat[1]) / 3])
        self.R = _rot_z(2 * np.pi / 3)

    def dynmat(self, q_crys):
        """Nearest-neighbour springs (longitudinal 1, transverse 0.3, z 0.2), unit masses, cell gauge."""
        q = 2 * np.pi * np.linalg.inv(self.lat) @ q_crys
        D = np.zeros((6, 6), complex)
        for Rb in (np.zeros(3), -self.lat[0], -self.lat[1]):
            d = self.tau[1] + Rb
            u = d / np.linalg.norm(d)
            K = np.outer(u, u) + 0.3 * (np.eye(3) - np.outer(u, u))
            K[2, 2] = 0.2
            D[:3, :3] += K; D[3:, 3:] += K
            D[:3, 3:] -= K * np.exp(1j * q @ Rb)
        D[3:, :3] = D[:3, 3:].conj().T
        w2, v = np.linalg.eigh(D)
        return w2, v.T.reshape(6, 2, 3)

    def test_chiral_modes_at_K(self):
        qK = np.array([1 / 3., -1 / 3., 0])
        w2, e = self.dynmat(qK)
        self.assertLess(abs(w2[0] - w2[1]), 1e-12)          # ZA/ZO degenerate at K
        j, W = phonon_tcam(e, w2, self.tau, self.lat, self.R, 3, q_crys=qK)
        np.testing.assert_allclose(np.sort(j), [-1, -1, 0, 0, 1, 1], atol=1e-8)
        np.testing.assert_allclose(W.conj().T @ W, np.eye(6), atol=1e-10)
        # degenerate ZA/ZO block is resolved into sublattice-pure Az (j=0) and Bz (j=-1)
        ez = np.einsum('ma,mkx->akx', W, e)[:2]
        for a in range(2):
            pol = np.abs(ez[a, :, 2])**2
            self.assertAlmostEqual(pol.max(), 1.0, places=8)
            self.assertAlmostEqual(j[a], 0 if pol[0] > 0.5 else -1, places=8)
        # 'atom' gauge e~ = e e^{-iq.tau} gives the same j
        q = 2 * np.pi * np.linalg.inv(self.lat) @ qK
        ea = e * np.exp(-1j * self.tau @ q)[None, :, None]
        ja, _ = phonon_tcam(ea, w2, self.tau, self.lat, self.R, 3, q_crys=qK, gauge='atom')
        np.testing.assert_allclose(ja, j, atol=1e-8)

    def test_gamma(self):
        """Gamma: in-plane E pairs j = +-1, z modes j = 0."""
        w2, e = self.dynmat(np.zeros(3))
        j, _ = phonon_tcam(e, w2, self.tau, self.lat, self.R, 3, degen_tol=1e-8)
        np.testing.assert_allclose(np.sort(j), [-1, -1, 0, 0, 1, 1], atol=1e-8)


class TestTCAMConservation(unittest.TestCase):

    def test_obeying_and_violating(self):
        rng = np.random.default_rng(0)
        j_in, j_out, j_ph = np.array([1, -1, 0]), np.array([1, -1, 0, 1]), np.array([0, 1, -1])
        G = rng.normal(size=(3, 3, 4)) + 1j * rng.normal(size=(3, 3, 4))
        dj = (j_out[None, None, :] - j_in[None, :, None] - j_ph[:, None, None]) % 3
        G[dj != 0] = 0
        frac, gmax, idx = check_tcam_conservation(G, j_out, j_in, j_ph, 3)
        self.assertEqual(frac, 0.0); self.assertIsNone(idx)
        # E' phonon (j=+1) couples sigma+ (j=+1) -> sigma- (j=-1): 1 + 1 = -1 mod 3
        self.assertNotEqual(G[1, 0, 1], 0)
        G[1, 0, 0] = 2.0          # sigma+ -> sigma+ with j_ph = +1: forbidden
        frac, gmax, idx = check_tcam_conservation(G, j_out, j_in, j_ph, 3)
        self.assertEqual(idx, (1, 0, 0)); self.assertAlmostEqual(gmax, 2.0)
        self.assertAlmostEqual(frac, 4.0 / np.sum(np.abs(G)**2))


if __name__ == '__main__':
    unittest.main()
