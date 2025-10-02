import sys
sys.path.insert(0, "/home/users/rreho/personalcodes/yambopy-project")

import numpy as np
import pytest

from yambopy.wannier.wann_model import TBMODEL


class DummyHR:
    def __init__(self, hop, HR_mn):
        self.hop = np.asarray(hop, float)  # (NR, 3) reduced coords
        self.HR_mn = np.asarray(HR_mn, complex)  # (NR, Nw, Nw)


class DummyLatDB:
    def __init__(self, lat):
        self.lat = np.asarray(lat, float)  # (3, 3) real-space lattice vectors


def build_dummy_model(hop, HR_mn, lat=None):
    # Create TBMODEL instance without calling base class __init__
    model = TBMODEL.__new__(TBMODEL)
    model.hr = DummyHR(hop, HR_mn)
    if lat is not None:
        model.latdb = DummyLatDB(lat)
    return model


def test_delta_R_from_tbmodel_pruning_and_cutoff():
    # R list includes: 0, e_x, e_y, 2 e_x
    hop = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
        ],
        dtype=float,
    )
    Nw = 2
    HR_mn = np.zeros((len(hop), Nw, Nw), dtype=complex)
    # Set amplitudes: strong for 0, e_x, e_y; tiny for 2 e_x (below tol)
    HR_mn[0, 0, 0] = 1.0
    HR_mn[1, 0, 1] = 0.5
    HR_mn[2, 1, 0] = -0.3j
    HR_mn[3, 0, 0] = 1e-10  # should be pruned by tol

    # Unit lattice (Angstrom), so |R| in Cart matches integer norm
    lat = np.eye(3)

    model = build_dummy_model(hop, HR_mn, lat=lat)

    Rh, Re = model.delta_R_from_tbmodel(tol=1e-8, R_cut=1.1, ensure_zero=True)

    # Expect to keep 0, e_x, e_y; drop 2 e_x by tol and/or cutoff
    expected = { (0,0,0), (1,0,0), (0,1,0) }
    got = set(map(tuple, Rh.tolist()))

    assert got == expected
    assert np.array_equal(Rh, Re)
    assert any((Rh == np.array([0,0,0])).all(axis=1))


def test_build_shells_tb_symmetric_and_gamma():
    # Only +e_x present in hr; symmetric=True should add -e_x and ensure Gamma
    hop = np.array([[1, 0, 0]], float)
    Nw = 1
    HR_mn = np.zeros((1, Nw, Nw), dtype=complex)
    HR_mn[0, 0, 0] = 1.0

    model = build_dummy_model(hop, HR_mn, lat=None)  # R_cut not used here

    Rh, Re = model.build_shells_for_bse(mode="tb", tol=1e-12, symmetric=True)

    have_e_x = any((Rh == np.array([1, 0, 0])).all(axis=1))
    have_m_e_x = any((Rh == np.array([-1, 0, 0])).all(axis=1))
    have_gamma = any((Rh == np.array([0, 0, 0])).all(axis=1))

    assert have_e_x and have_m_e_x and have_gamma
    # Electron shell mirrors hole shell in tb mode here
    assert set(map(tuple, Rh.tolist())) == set(map(tuple, Re.tolist()))


def test_delta_R_from_tbmodel_requires_lat_for_Rcut():
    # When R_cut is provided but no lattice is attached, expect ValueError
    hop = np.array([[0, 0, 0]], float)
    HR_mn = np.ones((1, 1, 1), dtype=complex)

    model = build_dummy_model(hop, HR_mn, lat=None)

    with pytest.raises(ValueError):
        _ = model.delta_R_from_tbmodel(tol=1e-8, R_cut=0.5, ensure_zero=True)