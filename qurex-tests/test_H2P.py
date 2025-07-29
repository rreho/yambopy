from yambopy.dbs.electronsdb import YamboElectronsDB
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.wannier.wann_io import HR
from yambopy.wannier.wann_model import TBMODEL
from yambopy.wannier.wann_mpgrid import tb_Monkhorst_Pack
from yambopy.wannier.coulombpot import CoulombPotentials
from yambopy.wannier.wann_H2p import H2P
import numpy as np
import tbmodels



def load_data():
    QE_PATH = './qurex-tests/data-tests/LiF/nscf-wannier-3x3x3'
    lat_k = YamboLatticeDB.from_db_file(f'{QE_PATH}/SAVE/', Expand=False)

    hrk = HR(f'{QE_PATH}/LiF')

    nnkp_kgrid = tb_Monkhorst_Pack([6,6,6], lat_k)
    nnkp_qgrid = tb_Monkhorst_Pack([3,3,3], lat_k)
    nnkp_kgrid.generate()
    nnkp_qgrid.generate()

    model = TBMODEL.from_wannier_files(
        hr_file=f'{QE_PATH}/LiF_hr.dat',
        win_file=f'{QE_PATH}/LiF.win',
    )
    model.set_mpgrid(nnkp_kgrid)

    model.solve_ham_from_hr(lat_k, hrk, fermie = 1)
    hlm = model.get_hlm(lat_k.lat, hrk)

    return {
        'lat_k': lat_k,
        'nnkp_kgrid': nnkp_kgrid,
        'nnkp_qgrid': nnkp_qgrid,
        'model': model,
        'hlm': hlm,
        'hrk': hrk,
    }
def load_coulomb_potentials(lat_k):
    cpot = CoulombPotentials(v0=0.0, lc=6, w=0.0, r0=1.0, ngrid=[6,6,6],
                             lattice=lat_k,ediel=[1,1,1],
                             tolr=0.001)
    return cpot

def build_h2p(model, nnkp_kgrid, nnkp_qgrid,cpot):
    QE_PATH = './qurex-tests/data-tests/LiF/nscf-wannier-3x3x3'

    savedb_path = f'{QE_PATH}/SAVE'
    h2p = H2P(model, savedb_path, kmpgrid=nnkp_kgrid,qmpgrid=nnkp_qgrid, kernel_path=None, method='model',ctype='v2dt2', cpot=cpot,bse_nv=1,bse_nc=1,
              run_parallel=True,ktype='NOIP',eta=0.10,gammaonly=False)
    h2p.solve_H2P()
    return h2p

def test_h2p():
    data = load_data()
    lat_k = data['lat_k']
    nnkp_kgrid = data['nnkp_kgrid']
    nnkp_qgrid = data['nnkp_qgrid']
    model = data['model']
    hlm = data['hlm']
    hrk = data['hrk']

    cpot = load_coulomb_potentials(lat_k)
    
    h2p = build_h2p(model, nnkp_kgrid, nnkp_qgrid, cpot)
    assert h2p.K_Ex is not None, "Exchange kernel is not computed, error in q-grid."
    assert h2p.h2peigv.shape == (27,216), "Shape of h2peigv is not as expected."
    assert h2p.h2peigvec_vck.shape == (27,216,1,1,216), "Shape of h2peigv_vck is not as expected."
    assert np.all(h2p.h2peigv > 0)