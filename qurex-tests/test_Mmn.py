import tbmodels
import numpy as np
from yambopy import YamboLatticeDB
from yambopy.wannier.wann_io import MMN, NNKP
from yambopy.dbs.wfdb import YamboWFDB
from yambopy.wannier.wann_Mssp import compute_overlap_kkpb


def load_wfdb():
    QE_PATH = './qurex-tests/data-tests/LiF/nscf-wannier-3x3x3/'
    lat_q = YamboLatticeDB.from_db_file(f'{QE_PATH}/SAVE/',Expand=False)
    lat_q.ibz_kpoints = lat_q.red_kpoints
    nnkp = NNKP(seedname=f'{QE_PATH}/LiF')
    mmn = MMN(seedname=f'{QE_PATH}/LiF')

    wfdb = YamboWFDB(path=f'{QE_PATH}/',latdb=lat_q, bands_range=[1,11])

    return wfdb, nnkp, mmn

def test_overlap_kkpb():
    """Since the SAVE is of a calculation in the full BZ, 
    we don't have to expand. There is no ibz in this case."""
    wfdb, nnkp, mmn = load_wfdb()
    wfdb.wf_bz = wfdb.wf
    wfdb.g_bz = wfdb.gvecs
    wfdb.ngBZ = wfdb.ngvecs
    Mkpb = compute_overlap_kkpb(wfdb, nnkp)
    assert np.count_nonzero(np.round(mmn.data - Mkpb,10)) == 0

