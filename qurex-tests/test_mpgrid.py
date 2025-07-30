import numpy as np
from yambopy import YamboLatticeDB, tb_Monkhorst_Pack

def load_grids():
    WORK_PATH = './qurex-tests/data-tests/LiF/'
    QE_PATH = './qurex-tests/data-tests/LiF/'
    YAMBO_PATH = './qurex-tests/data-tests/LiF/Optics'
    lat_k = YamboLatticeDB.from_db_file(f'{YAMBO_PATH}/yambo-DS/10x10x10/SAVE/')
    # lat_q = YamboLatticeDB.from_db_file(f'{YAMBO_PATH}/yambo-DS/5x5x5/SAVE/')
    mp_kgrid = tb_Monkhorst_Pack([6,6,6],lat_k)
    mp_qgrid = tb_Monkhorst_Pack([3,3,3],lat_k)
    mp_kgrid.generate()
    mp_qgrid.generate()

    return mp_kgrid, mp_qgrid


def test_grids():
    mp_kgrid, mp_qgrid = load_grids()
    (kplusq_table, kminusq_table) = mp_kgrid.get_kq_tables(mp_qgrid) 
    (qplusk_table, qminusk_table) = mp_qgrid.get_kq_tables(mp_kgrid, sign="-")  # minus sign to have k-q  
    ref_kplusq = np.load("qurex-tests/data-tests/ref-data/mp_kpq_grid_table.npy")
    ref_kminusq = np.load("qurex-tests/data-tests/ref-data/mp_kmq_grid_table.npy")
    ref_qplusk = np.load("qurex-tests/data-tests/ref-data/mp_qpk_grid_table.npy")
    ref_qminusk = np.load("qurex-tests/data-tests/ref-data/mp_qmk_grid_table.npy")

    assert np.allclose(kplusq_table, ref_kplusq, atol=1e-8)
    assert np.allclose(kminusq_table, ref_kminusq, atol=1e-8)
    assert np.allclose(qplusk_table, ref_qplusk, atol=1e-8)
    assert np.allclose(qminusk_table, ref_qminusk, atol=1e-8)    
