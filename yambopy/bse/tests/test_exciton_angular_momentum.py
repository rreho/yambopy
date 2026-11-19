# Copyright (c) 2026, University of Luxembourg
# All rights reserved.
#
# This file is part of yambopy
#
import unittest
import numpy as np
from yambopy.bse.circular_basis import E_PLUS, E_MINUS
from yambopy.bse.exciton_angular_momentum import (tcam_from_rotation,
                                                  find_rotation, lz_density)


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


if __name__ == '__main__':
    unittest.main()
