import sys
sys.path.insert(0, "/home/users/rreho/personalcodes/yambopy-project")

import numpy as np
from numpy.linalg import norm
import pytest

from yambopy.wannier.wann_model import TBMODEL
from yambopy.wannier.wann_bse_wannier import BSEWannierFT
from yambopy.wannier.wann_mpgrid import tb_Monkhorst_Pack


class DummyLatDB:
    def __init__(self, lat=None, rlat=None):
        self.lat = np.eye(3) if lat is None else np.asarray(lat, float)
        self.rlat = np.eye(3) if rlat is None else np.asarray(rlat, float)
        # Minimal attributes used elsewhere (not required here)
        self.red_atomic_positions = None
        self.atomic_numbers = None


class DummyHR:
    def __init__(self, hop, HR_mn, ws_deg):
        self.hop = np.asarray(hop, float)       # (NR,3) reduced
        self.HR_mn = np.asarray(HR_mn, complex) # (NR,Nw,Nw)
        self.ws_deg = np.asarray(ws_deg, int)   # (NR,)
        self.num_wann = HR_mn.shape[1]
        self.nrpts = self.hop.shape[0]


def build_minimal_tb_model(nkx=2, nky=2, nkz=1):
    # Lattice (identity)
    latdb = DummyLatDB()
    # HR: two orbitals, only on-site terms -> k-independent diagonal bands -1,+1
    hop = np.array([[0, 0, 0]], float)
    HR_mn = np.zeros((1, 2, 2), complex)
    HR_mn[0, 0, 0] = -1.0
    HR_mn[0, 1, 1] = +1.0
    ws_deg = np.array([1], int)
    hr = DummyHR(hop, HR_mn, ws_deg)

    # Build k-grid
    mpgrid = tb_Monkhorst_Pack((nkx, nky, nkz), latdb)
    mpgrid.generate()

    # TB model instance and setup
    model = TBMODEL.__new__(TBMODEL)
    TBMODEL.set_mpgrid(mpgrid)
    # Solve from hr with Fermi level between bands to get Nv=1, Nc=1
    model.solve_ham_from_hr(latdb=latdb, hr=hr, fermie=0.0)
    return model


def test_bse_wannier_tb_roundtrip_simple(seed=123):
    np.random.seed(seed)

    # 1) Minimal TB model and shells from the dual grid (exact inverse)
    model = build_minimal_tb_model(nkx=2, nky=2, nkz=1)
    Rh, Re = model.build_shells_for_bse(mode="dual", symmetric=False)
    assert Rh.shape[1] == 3 and Re.shape[1] == 3

    # 2) Get U_val, U_cond from model (Nv=Nc=1 here)
    U_val, U_cond = model.get_U_val_cond()
    Nk = model.mpgrid.nkpoints
    Nv = model.nv
    Nc = model.nc
    assert Nv == 1 and Nc == 1

    # 3) Build transformer using existing kpoints/Q and generated shells
    kpoints = model.mpgrid.k
    Q = np.array([0.0, 0.0, 0.0])
    ft = BSEWannierFT(kpoints=kpoints, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Rh, Se_list=Re)

    # 4) Construct a Bloch kernel K(k,k') that is exactly recovered with TB shells:
    #    Let K(k, k+q) = alpha * F(q) with alpha = Nk/(NRh*NRe), so that inverse scaling matches.
    NRh, NRe = Rh.shape[0], Re.shape[0]
    alpha = Nk / float(NRh * NRe)
    Fq = np.random.randn(Nk)  # real values per q-index

    K6 = np.zeros((Nk, Nk, Nv, Nc, Nv, Nc), dtype=complex)
    for ik in range(Nk):
        for iq in range(Nk):
            ikp = ft.kpq[ik, iq]
            K6[ik, ikp, 0, 0, 0, 0] = alpha * Fq[iq]

    # 5) Forward and inverse FT
    Wtilde = ft.bloch_to_wtilde(K6)
    K6_back = ft.wtilde_to_bloch(Wtilde)

    rel_err = norm((K6_back - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
    print("TB shells round-trip relative error:", rel_err)

    # With the construction above, this should be machine-precision exact
    assert rel_err < 1e-12


@pytest.mark.parametrize("nkx,nky,nkz", [(2,2,1), (4,4,1), (6,6,1)])
def test_bse_wannier_dual_roundtrip_param(nkx, nky, nkz, seed=123):
    rng = np.random.RandomState(seed)

    # Build TB model and dual shells (no symmetrization) for exact inversion
    model = build_minimal_tb_model(nkx=nkx, nky=nky, nkz=nkz)
    Rh, Re = model.build_shells_for_bse(mode="dual", symmetric=False)
    U_val, U_cond = model.get_U_val_cond()

    Nk = model.mpgrid.nkpoints
    Nv, Nc = model.nv, model.nc
    assert Nv == 1 and Nc == 1

    kpoints = model.mpgrid.k

    # Choose a few on-grid Q vectors: Gamma, first step in kx (if available), first step in ky (if available)
    iqs = [0]
    if Nk > 1:
        iqs.append(1)
    # add first step along y if present (index nkx)
    if hasattr(model.mpgrid, 'nkx') and Nk > model.mpgrid.nkx:
        iqs.append(model.mpgrid.nkx)

    for iq_choice in iqs:
        Q = kpoints[iq_choice]
        ft = BSEWannierFT(kpoints=kpoints, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Rh, Se_list=Re)

        NRh, NRe = Rh.shape[0], Re.shape[0]
        alpha = Nk / float(NRh * NRe)
        Fq = rng.randn(Nk)

        K6 = np.zeros((Nk, Nk, Nv, Nc, Nv, Nc), dtype=complex)
        for ik in range(Nk):
            for iq in range(Nk):
                ikp = ft.kpq[ik, iq]
                K6[ik, ikp, 0, 0, 0, 0] = alpha * Fq[iq]

        Wtilde = ft.bloch_to_wtilde(K6)
        K6_back = ft.wtilde_to_bloch(Wtilde)

        rel_err = norm((K6_back - K6).reshape(-1)) / max(1.0, norm(K6.reshape(-1)))
        assert rel_err < 1e-12, f"Grid {(nkx,nky,nkz)} Q_idx {iq_choice} rel_err {rel_err}"