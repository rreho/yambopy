#
# License-Identifier: GPL
#
# Copyright (C) 2026 University of Luxembourg
#
# This file is part of the yambopy project
#
import unittest
import numpy as np
from yambopy.units import ha2ev
from yambopy.tools.funcs import boltzman_f
from yambopy.exciton_phonon.excph_luminescence import (exc_ph_luminescence, exc_direct_luminescence,
                                                   exc_ph_renormalisation, exc_ph_zpl_depletion)
from yambopy.tools.funcs import bose


def _inputs(seed=3, nq=4, nm=6, nin=3, nout=4):
    rng = np.random.default_rng(seed)
    ph  = 0.02 + 0.15*rng.random((nq, nm))
    Eo  = 2.0 + 0.3*rng.random((nq, nout))
    Ei  = 2.0 + 0.3*rng.random(nin)
    D   = rng.normal(size=(3, nin)) + 1j*rng.normal(size=(3, nin))
    G   = (rng.normal(size=(nq, nm, nin, nout)) + 1j*rng.normal(size=(nq, nm, nin, nout)))*1e-3
    return ph, Eo, Ei, D, G


class TestExcPhLuminescence(unittest.TestCase):

    kw = dict(emin=1.5, emax=2.5, estep=0.005, broad=0.01)

    def test_cutoffs_off_by_default(self):
        """omega_min=0 and a vanishing res_min must reproduce the original formula."""
        ph, Eo, Ei, D, G = _inputs()
        _, ref = exc_ph_luminescence(30, ph, Eo, D, G, exc_energies_in=Ei, **self.kw)
        _, a   = exc_ph_luminescence(30, ph, Eo, D, G, exc_energies_in=Ei, omega_min=0., res_min=1e-12, **self.kw)
        np.testing.assert_allclose(a, ref, rtol=1e-12)

    def test_res_min_removes_onshell_term(self):
        """A finite-q state resonant with E_in + omega must be removed by res_min only."""
        ph, Eo, Ei, D, G = _inputs()
        Ei[0] = Eo.min() - ph[1, 2]          # make the resonant final state the lowest (populated)
        Eo[1, 0] = Ei[0] + ph[1, 2]          # exact double pole (emission)
        _, a = exc_ph_luminescence(30, ph, Eo, D, G, exc_energies_in=Ei, broad_0=1e-3, **self.kw)
        _, b = exc_ph_luminescence(30, ph, Eo, D, G, exc_energies_in=Ei, broad_0=1e-3, res_min=1e-4, **self.kw)
        _, c = exc_ph_luminescence(30, ph, Eo, D, G, exc_energies_in=Ei, broad_0=2e-3, res_min=1e-4, **self.kw)
        self.assertGreater(a.max(), 10*b.max())          # artefact dominated without cutoff
        # with the cutoff only off-resonant terms remain: eta-dependence O(eta^2/Delta^2) ~ 1%
        np.testing.assert_allclose(b, c, rtol=5e-2, atol=1e-6*b.max())

    def test_omega_min(self):
        ph, Eo, Ei, D, G = _inputs()
        G2 = G.copy(); G2[ph < 0.05] = 0.
        _, a = exc_ph_luminescence(30, ph, Eo, D, G,  exc_energies_in=Ei, omega_min=0.05, **self.kw)
        _, b = exc_ph_luminescence(30, ph, Eo, D, G2, exc_energies_in=Ei, **self.kw)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_direct_area(self):
        """Integrated direct PL = sum_l |D_l|^2/3 N_l / E_l(Ha) * ha2ev**3 (same units as satellites)."""
        _, _, Ei, D, _ = _inputs()
        w, I = exc_direct_luminescence(30, Ei, D, emin=0., emax=5., estep=1e-4, broad=0.002)
        occ  = boltzman_f(Ei-Ei.min(), 30)
        ref  = np.sum(np.sum(np.abs(D)**2, axis=0)/3.*occ*ha2ev/Ei)*ha2ev**3  # (1/E_Ha) * Lorentzian area in eV * ha2ev**2
        self.assertAlmostEqual(np.trapezoid(I, w)/ref, 1., delta=2e-3)

    def test_sum_rule(self):
        """(1-R) direct + satellites = bare direct weight (one Q=0 state, flat populations,
        no energy prefactor, eta_0 -> 0): R is exactly the satellite weight."""
        ph, Eo, Ei, D, G = _inputs(nin=1)
        Ei[0] = Eo.min() - 0.05                       # no near-resonant terms
        kw = dict(emin=0., emax=5., estep=2e-4, broad=0.002)
        Texc = 1e9                                    # flat Boltzmann populations
        _, d0  = exc_direct_luminescence(Texc, Ei, D, PL_energy_prefactor='', **kw)
        for eta in (1e-9, 5e-3):                      # same regularisation in R and satellites
            R = exc_ph_renormalisation(30, ph, Eo, G, exc_energies_in=Ei, broad_0=eta)
            w, sat = exc_ph_luminescence(30, ph, Eo, D, G, exc_energies_in=Ei, exc_temp=Texc,
                                         PL_energy_prefactor='', broad_0=eta, **kw)
            _, dR  = exc_direct_luminescence(Texc, Ei, D, PL_energy_prefactor='', R_lambda=R, **kw)
            self.assertGreater(R[0], 1e-4)
            self.assertAlmostEqual(np.trapezoid(sat, w)/np.trapezoid(d0, w), R[0], delta=2e-3*R[0])
            self.assertAlmostEqual(np.trapezoid(dR+sat, w)/np.trapezoid(d0, w), 1., delta=1e-3)

class TestZPLDepletion(unittest.TestCase):
    """Two-level toy: Q=0 state l, one finite-q state b, one mode w, one q."""
    El, Eb, w, g = 4.10, 4.25, 0.17, 0.004        # eV (g in eV)

    def args(self, T):
        G = np.array([[[[self.g/ha2ev]]]])
        return (T, np.array([[self.w]]), np.array([[self.Eb]]), G), dict(exc_energies_in=np.array([self.El]))

    def test_exact_two_level_T0(self):
        """T=0: |l> couples to |b + 1 phonon>; Z = |<l|psi>|^2 of the exact 2x2 eigenstate."""
        a, kw = self.args(0.)
        H = np.array([[self.El, self.g], [self.g, self.Eb + self.w]])
        e, v = np.linalg.eigh(H)
        Z = np.abs(v[0, np.argmax(np.abs(v[0]))])**2
        dZ = exc_ph_zpl_depletion(*a, **kw)[0]
        self.assertAlmostEqual(dZ, self.g**2/(self.El - self.Eb - self.w)**2, places=12)
        self.assertAlmostEqual(dZ, 1 - Z, delta=5*dZ**2)
        # replica weight pairs n+1 with E_l-E_b+w instead
        R = exc_ph_renormalisation(*a, **kw)[0]
        self.assertAlmostEqual(R, self.g**2/(self.El - self.Eb + self.w)**2, places=12)

    def test_derivative_of_sigma(self):
        """1-Z = -dRe Sigma/dw at E_l, with eta and phonon occupation."""
        T, eta = 300., 0.01
        a, kw = self.args(T)
        n = bose(np.array([self.w]), T)[0]
        sig = lambda x: np.real(self.g**2*((n + 1)/(x - self.Eb - self.w + 1j*eta) + n/(x - self.Eb + self.w + 1j*eta)))
        h = 1e-6
        ref = -(sig(self.El + h) - sig(self.El - h))/(2*h)
        dZ = exc_ph_zpl_depletion(*a, broad_0=eta, **kw)[0]
        self.assertAlmostEqual(dZ, ref, delta=1e-7*abs(ref))
        # res_min drops the near-resonant term (|E_l-E_b+w| = 20 meV)
        dZ1 = exc_ph_zpl_depletion(*a, broad_0=eta, res_min=0.03, **kw)[0]
        x = self.El - self.Eb - self.w
        self.assertAlmostEqual(dZ1, self.g**2*(n + 1)*(x**2 - eta**2)/(x**2 + eta**2)**2, places=12)


class TestDirectZPLWeights(unittest.TestCase):

    def test_zpl_depletion_weights_and_cutoff(self):
        E = np.array([2.0, 2.5, 3.0]); D = np.ones((3, 3), complex)
        kw = dict(emin=1.5, emax=3.5, estep=0.001, broad=0.01, exc_min_energy=2.0)
        one = [exc_direct_luminescence(300., E[[l]], D[:, [l]], **kw)[1] for l in range(3)]
        _, I1 = exc_direct_luminescence(300., E, D, zpl_depletion=[0.1, 0.5, 0.3], **kw)
        # Z = 0.9 kept; 1-Z = 0.5 > 0.3 omitted; 1-Z = 0.3 <= 0.3 kept with Z = 0.7
        np.testing.assert_allclose(I1, 0.9*one[0] + 0.7*one[2], rtol=1e-6, atol=1e-12*one[0].max())
        _, I2 = exc_direct_luminescence(300., E, D, zpl_depletion=[0.1, 0.5, 0.3], max_depletion=1., **kw)
        np.testing.assert_allclose(I2, 0.9*one[0] + 0.5*one[1] + 0.7*one[2], rtol=1e-6, atol=1e-12*one[0].max())

if __name__ == '__main__':
    unittest.main()
