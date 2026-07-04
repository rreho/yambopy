# Copyright (c) 2026, University of Luxembourg
# All rights reserved.
#
# This file is part of yambopy
#
import unittest
import numpy as np
from yambopy.bse.circular_basis import circular_basis_matrix, E_PLUS, E_MINUS
from yambopy.dbs.excitondb import YamboExcitonDB


def _purity(D):
    pR = abs(np.vdot(E_PLUS, D))**2
    pL = abs(np.vdot(E_MINUS, D))**2
    return max(pR, pL) / (pR + pL)


class TestCircularBasis(unittest.TestCase):

    def test_random_mixture_recovers_pure_states(self):
        """A randomly mixed sigma+/sigma- doublet must rotate back to 100% purity."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            th = rng.uniform(0, np.pi/2)
            ph1, ph2 = rng.uniform(0, 2*np.pi, 2)
            V = np.array([[np.cos(th)*np.exp(1j*ph1), -np.sin(th)*np.exp(1j*ph2)],
                          [np.sin(th)*np.exp(-1j*ph2), np.cos(th)*np.exp(-1j*ph1)]])
            D_raw = np.stack([E_PLUS, E_MINUS], axis=1) @ V
            E = np.array([1.0, 1.0 + 1e-6])

            U, blocks = circular_basis_matrix(E, D_raw, atol=1e-4)
            self.assertEqual(len(blocks), 1)
            # unitarity
            np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-12)
            D_rot = D_raw @ U
            # column 0 is sigma+, column 1 is sigma-, both fully pure
            self.assertGreater(abs(np.vdot(E_PLUS, D_rot[:, 0]))**2, 0.999999)
            self.assertGreater(abs(np.vdot(E_MINUS, D_rot[:, 1]))**2, 0.999999)
            for i in range(2):
                self.assertGreater(_purity(D_rot[:, i]), 1 - 1e-12)

    def test_nondegenerate_and_dark_states_untouched(self):
        """Non-degenerate states and dark degenerate blocks must not be rotated."""
        D = np.zeros((3, 4), dtype=complex)
        D[:, 0] = E_PLUS                 # bright, non-degenerate
        D[:, 2] = 1e-6 * E_PLUS          # dark doublet (numerical noise dipoles)
        D[:, 3] = 1e-6 * E_MINUS
        E = np.array([1.0, 1.5, 2.0, 2.0 + 1e-6])
        U, blocks = circular_basis_matrix(E, D, atol=1e-4)
        self.assertEqual(len(blocks), 0)
        np.testing.assert_allclose(U, np.eye(4), atol=1e-15)

    def test_flatten_unflatten_roundtrip(self):
        """Regression for the flatten_Akcv scatter/gather bug (commit 41e33e33):
        flatten_Akcv must be the exact inverse of get_Akcv."""
        rng = np.random.default_rng(3)
        nk, ncb, nvb, neigs = 5, 3, 2, 4
        ntrans = nk*ncb*nvb

        # BS table in a scrambled row order, as yambo may deliver it
        rows = [(ik+1, iv+1, ic+nvb+1, 1, 1)
                for ik in range(nk) for ic in range(ncb) for iv in range(nvb)]
        table = np.array(rows)[rng.permutation(ntrans)]

        class Dummy:
            pass
        db = Dummy()
        db.table = table
        db.nkpoints, db.ncbands, db.nvbands = nk, ncb, nvb
        db.spin_pol = 'no'
        db.eigenvectors = (rng.standard_normal((neigs, ntrans))
                           + 1j*rng.standard_normal((neigs, ntrans)))
        db.Akcv = None

        Akcv = YamboExcitonDB.get_Akcv(db)
        self.assertEqual(Akcv.shape, (neigs, 1, 1, nk, ncb, nvb))
        flat = YamboExcitonDB.flatten_Akcv(db, Akcv)
        np.testing.assert_allclose(flat, db.eigenvectors, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
