# tests/test_kgrid_tables.py

import numpy as np
from yambopy import YamboLatticeDB, NNKP_Grids

def load_grids():
    WORK_PATH = './qurex-tests/data-tests/LiF/'
    QE_PATH = './qurex-tests/data-tests/LiF/'
    YAMBO_PATH = './qurex-tests/data-tests/LiF/Optics'

    lat_k = YamboLatticeDB.from_db_file(f'{YAMBO_PATH}/yambo-DS/10x10x10/SAVE/')
    lat_q = YamboLatticeDB.from_db_file(f'{YAMBO_PATH}/yambo-DS/5x5x5/SAVE/')
    nnkp_kgrid = NNKP_Grids(f'{QE_PATH}/nscf-wannier-10x10x10//LiF')
    nnkp_qgrid = NNKP_Grids(f'{QE_PATH}/nscf-wannier-5x5x5//LiF')

    nnkp_kgrid.get_kmq_grid(nnkp_qgrid)
    # nnkp_kgrid.get_qpb_grid(nnkp_kgrid)
    nnkp_qgrid.get_qpb_grid(nnkp_qgrid)
    nnkp_kgrid.get_kpbover2_grid(nnkp_kgrid)
    nnkp_kgrid.get_kmqmbover2_grid(nnkp_qgrid)

    return nnkp_kgrid

def test_kmq_grid():
    nnkp = load_grids()
    result = nnkp.kmq_grid_table
    expected = np.load("qurex-tests/data-tests/ref-data/kmq_grid.npy")
    assert np.allclose(result, expected, atol=1e-8)

def test_kpbover2_grid():
    nnkp = load_grids()
    result = nnkp.kpbover2_grid_table
    expected = np.load("qurex-tests/data-tests/ref-data/kpbover2_grid.npy")
    assert np.allclose(result, expected, atol=1e-8)

def test_kmqmbover2_grid():
    nnkp = load_grids()
    result = nnkp.kmqmbover2_grid_table
    expected = np.load("qurex-tests/data-tests/ref-data/kmqmbover2_grid.npy")
    assert np.allclose(result, expected, atol=1e-8)
