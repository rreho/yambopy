# tests/test_kgrid_tables.py

import numpy as np
from yambopy import YamboLatticeDB, NNKP_Grids

def load_grids():
    WORK_PATH = './qurex-tests/data-tests/LiF/'
    QE_PATH = './qurex-tests/data-tests/LiF/'
    YAMBO_PATH = './qurex-tests/data-tests/LiF/Optics'

    nnkp_kgrid = NNKP_Grids(f'{QE_PATH}/nscf-wannier-10x10x10//LiF')
    nnkp_qgrid = NNKP_Grids(f'{QE_PATH}/nscf-wannier-5x5x5//LiF')

    nnkp_kgrid.get_kmq_grid(nnkp_qgrid,sign='-')
    nnkp_kgrid.get_kpb_grid(nnkp_kgrid)
    nnkp_qgrid.get_qpb_grid(nnkp_qgrid)

    return nnkp_kgrid, nnkp_qgrid

def test_kpb_grid():
    '''k + b or k + B/2'''
    nnkp_kgrid, nnkp_qgrid = load_grids()
    result = nnkp_kgrid.kpb_grid_table
    reference = np.take_along_axis(nnkp_kgrid.nnkp, nnkp_kgrid.sort_idx[:,:,None], axis=1)
    assert np.allclose(result, reference, atol=1e-8)

def test_kmb_grid():
    '''k - b or k - B/2'''
    nnkp_kgrid, nnkp_qgrid = load_grids()
    result = nnkp_kgrid.kmb_grid
    reference = nnkp_kgrid.kpb_grid[:,::-1,:] # Because the b-vecs are sorted this works
    assert np.allclose(result, reference, atol=1e-8)

def test_kmq_grid():
    '''k - q'''
    nnkp_kgrid, nnkp_qgrid = load_grids()
    result = nnkp_kgrid.kmq_grid
    reference = np.load("qurex-tests/data-tests/ref-data/nnkp_kmq_grid.npy")
    assert np.allclose(result, reference, atol=1e-8)

def test_qmb_grid():
    '''q - B'''
    nnkp_kgrid, nnkp_qgrid = load_grids()
    result = nnkp_qgrid.qmb_grid
    reference = nnkp_qgrid.qpb_grid[:,::-1,:] # Because the b-vecs are sorted this works
    assert np.allclose(result, reference, atol=1e-8)

def test_qpb_grid():
    '''q + B'''
    nnkp_kgrid, nnkp_qgrid = load_grids()
    result = nnkp_qgrid.qpb_grid_table
    reference = np.take_along_axis(nnkp_qgrid.nnkp, nnkp_qgrid.sort_idx[:,:,None], axis=1)
    assert np.allclose(result, reference, atol=1e-8)
