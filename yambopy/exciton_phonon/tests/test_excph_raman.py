#
# Authors: RR
#
# Tests for yambopy.exciton_phonon.excph_raman
#
import os
import sys
import importlib.util
import tempfile
import unittest
import numpy as np
from yambopy.units import ha2ev
from yambopy.exciton_phonon.excph_raman import exc_ph_raman, exc_ph_raman_intensity

PHDSCRIPTS_EXPH = '/Users/riccardo.reho/codes/PhdScripts/exph'


def _load_phdscripts_raman():
    """Import PhdScripts/exph/raman.py by path (needs numba and its sibling modules)."""
    # keep numba/bytecode caches out of the (read-only) source tree
    os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp(prefix='numba_raman_'))
    sys.dont_write_bytecode = True
    path = os.path.join(PHDSCRIPTS_EXPH, 'raman.py')
    if PHDSCRIPTS_EXPH not in sys.path: sys.path.insert(0, PHDSCRIPTS_EXPH)
    spec = importlib.util.spec_from_file_location('phd_raman', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['phd_raman'] = mod   # numba needs to re-import the module
    spec.loader.exec_module(mod)
    return mod


class TestExcPhRaman(unittest.TestCase):

    @unittest.skipUnless(os.path.isfile(os.path.join(PHDSCRIPTS_EXPH, 'raman.py')),
                         'PhdScripts raman.py not available')
    def test_against_phdscripts(self):
        try:
            phd = _load_phdscripts_raman()
        except ImportError as err:
            self.skipTest(f'cannot import PhdScripts raman.py: {err}')
        rng = np.random.default_rng(1234)
        nexc, nmodes = 7, 4
        exc_eV = np.sort(rng.uniform(1.5, 2.5, nexc))
        ph_eV = np.array([0.0002, 0.015, 0.025, 0.036])   # first mode below 5 cm-1 threshold
        dip = rng.normal(size=(3, nexc)) + 1j * rng.normal(size=(3, nexc))
        G = rng.normal(size=(nmodes, nexc, nexc)) + 1j * rng.normal(size=(nmodes, nexc, nexc))
        laser = np.linspace(1.4, 2.6, 23)
        broad, nk, vol = 0.08, 36, 312.7

        R = exc_ph_raman(ph_eV, exc_eV, dip, G, laser, broadening=broad,
                         cell_vol=vol, nkpts=nk, ph_fre_th=5.0)
        # PhdScripts: exciton/phonon energies in Ha, laser and broadening in eV with 27.211
        c = 27.211 / ha2ev
        R_ref = phd.compute_Raman_oneph_exc(laser * c, ph_eV / ha2ev, exc_eV / ha2ev, dip, G,
                                            nk, vol, broading=broad * c, npol=3,
                                            ph_fre_th=5, precision='d')
        self.assertEqual(R.shape, (len(laser), nmodes, 3, 3))
        self.assertTrue(np.all(R[:, 0] == 0))
        rel = np.max(np.abs(R - R_ref)) / np.max(np.abs(R_ref))
        self.assertLess(rel, 1e-10, msg=f'max rel. diff {rel:.2e}')

    def test_symmetric_mode_isotropic_dipoles(self):
        # Excitons in degenerate triplets (x,y,z); fully symmetric mode couples
        # S_t,i -> S_t',j only for i=j with i-independent amplitude  ->  R diagonal and isotropic
        rng = np.random.default_rng(7)
        ntrip = 4
        nexc = 3 * ntrip
        exc_eV = np.repeat(np.linspace(1.8, 2.4, ntrip), 3)
        dip = np.zeros((3, nexc), dtype=complex)
        amp = rng.normal(size=ntrip) + 1j * rng.normal(size=ntrip)
        for t in range(ntrip):
            for i in range(3): dip[i, 3 * t + i] = amp[t]
        g = rng.normal(size=(ntrip, ntrip)) + 1j * rng.normal(size=(ntrip, ntrip))
        g = g + g.conj().T                                  # hermitian at q=0
        G = np.kron(g, np.eye(3))[None]                     # [1, nexc, nexc]
        laser = np.linspace(1.6, 2.6, 11)
        R = exc_ph_raman(np.array([0.03]), exc_eV, dip, G, laser, broadening=0.05)[:, 0]
        off = R - np.einsum('wii->wi', R)[:, :, None] * np.eye(3)
        self.assertLess(np.max(np.abs(off)), 1e-12 * np.max(np.abs(R)))
        self.assertTrue(np.allclose(R[:, 0, 0], R[:, 1, 1]) and np.allclose(R[:, 0, 0], R[:, 2, 2]))
        I_xx = exc_ph_raman_intensity(R[:, None], e_in=[1, 0, 0])
        I_xy = exc_ph_raman_intensity(R[:, None], e_in=[1, 0, 0], e_out=[0, 1, 0])
        self.assertTrue(np.allclose(I_xx[:, 0], np.abs(R[:, 0, 0])**2))
        self.assertLess(np.max(I_xy), 1e-20 * np.max(I_xx))


if __name__ == '__main__':
    unittest.main()
