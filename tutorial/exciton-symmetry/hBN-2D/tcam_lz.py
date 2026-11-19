#
# Exciton symmetry analysis of monolayer hBN at Q = Gamma:
#   irreps -> circular (sigma+/-) basis -> TCAM j (C3) -> hole-centred L_z cubes
#
# Usage:  python tcam_lz.py [calc_dir] [bse_dir] [out_dir]
#   calc_dir : folder containing SAVE/            (default '.')
#   bse_dir  : folder with ndb.BS_diago_Q1 and ndb.dipoles, Lkind="bar"
#   out_dir  : where the cube files are written   (default '.')
#
import os, sys
import numpy as np
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.dbs.wfdb import YamboWFDB
from yambopy.symmetries.crystal_symmetries import Crystal_Symmetries
from yambopy.tools.degeneracy_finder import find_degeneracy_evs
from yambopy.bse.exciton_irreps import compute_exc_rep
from yambopy.bse.excitondipoles import exc_dipoles_pol
from yambopy.bse.circular_basis import rotate_excdb_to_circular, E_PLUS, E_MINUS
from yambopy.bse.realSpace_excitonwf import ex_wf2Real
from yambopy.io.cubetools import write_cube
from yambopy.bse.exciton_angular_momentum import (find_rotation, exciton_rotation_rep,
                                                  tcam_from_rotation, write_lz_cube)

path    = sys.argv[1] if len(sys.argv) > 1 else '.'
bse_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(path, 'bse_bar')
out_dir = sys.argv[3] if len(sys.argv) > 3 else '.'
nstates   = 10      # lowest excitons to analyse (complete multiplets!)
degen_tol = 1e-2    # eV
supercell = [7, 7, 1]
cutoff_Ry = 40.0

# 1) Irreducible representations (D3h) of the exciton multiplets
compute_exc_rep(path=path, bse_dir=bse_dir, iqpt=1, nstates=nstates,
                degen_tol=degen_tol, symm_tol=0.06)

# 2) Circular basis inside each degenerate bright block
lat = YamboLatticeDB.from_db_file(os.path.join(path, 'SAVE', 'ns.db1'))
exc = YamboExcitonDB.from_db_file(lat, filename='ndb.BS_diago_Q1', folder=bse_dir)
dip = exc_dipoles_pol(os.path.join(path, 'SAVE'), dipoles_path=bse_dir,
                      bse_path=bse_dir, save_files=False)        # (3, nexc), emission
U, blocks, dip = rotate_excdb_to_circular(exc, dip, atol=degen_tol, verbose=True)
E = exc.eigenvalues.real

# 3) TCAM from the C3 (counter-clockwise, about z) representation matrix
wfdb = YamboWFDB(path=path, latdb=lat,
                 bands_range=[np.min(exc.table[:, 1]) - 1, np.max(exc.table[:, 2])])
symm = Crystal_Symmetries(lat, tol=1e-4)
ic3 = find_rotation(symm.rotations, 3)
R3, t3 = symm.rotations[ic3], symm.translations[ic3]
dmats = wfdb.Dmat(symm_mat=R3[None], frac_vec=t3[None], time_rev=False)[0]
Ak = exc.get_Akcv()[:nstates]
D3 = exciton_rotation_rep(Ak, R3, wfdb, dmats, tau_cart=t3)

sp = np.abs(dip.T @ np.conj(E_PLUS))**2
sm = np.abs(dip.T @ np.conj(E_MINUS))**2
print("\n  S    E (eV)   |D|^2/max  sigma+ frac  <C3> (|.|, deg)   j(diag)")
s2 = (np.abs(dip[:, :nstates])**2).sum(0)
for n in range(nstates):
    l = D3[n, n]
    jd = -3 * np.angle(l) / (2 * np.pi)
    fr = sp[n] / (sp[n] + sm[n]) if sp[n] + sm[n] > 1e-8 * s2.max() else np.nan
    print(" %3d  %8.4f  %8.4f   %8.3f    (%.3f, %7.1f)  %+5.2f"
          % (n + 1, E[n], s2[n] / s2.max(), fr, abs(l), np.degrees(np.angle(l)), jd))
print("\nTCAM per multiplet (eigenvalues of D(C3) in the block):")
for blk in find_degeneracy_evs(E[:nstates], atol=degen_tol):
    blk = np.sort(blk)
    j, lam, _ = tcam_from_rotation(D3[np.ix_(blk, blk)], 3)
    print("  states %-12s E=%.4f eV  j = %s" % ((blk + 1).tolist(), E[blk].mean(),
                                                 np.round(j, 3).tolist()))

# 4) Real-space wavefunction (hole 1 bohr above N) and L_z density cubes
iN = int(np.where(lat.atomic_numbers == 7)[0][0])
hole = np.linalg.solve(lat.lat.T, lat.car_atomic_positions[iN]) % 1
hole[2] += 1.0 / lat.lat[2, 2]
bright = [n for n in range(nstates) if s2[n] > 1e-2 * s2.max()][:2]
scl, at_nums, at_pos, wf = ex_wf2Real(Ak[bright], np.zeros(3), wfdb,
                                      [np.min(exc.table[:, 1]), np.max(exc.table[:, 2])],
                                      hole, fix_particle='h', supercell=supercell,
                                      wfcCutoffRy=cutoff_Ry)
os.makedirs(out_dir, exist_ok=True)
for i, n in enumerate(bright):
    lab = 'S%d_%s_%.4feV' % (n + 1, 'sigma_plus' if sp[n] > sm[n] else 'sigma_minus', E[n])
    dens = (np.abs(wf[i])**2).sum(axis=(0, 1, 2))
    write_cube(os.path.join(out_dir, 'exc_%s_density.cube' % lab), dens / dens.max(),
               scl, at_pos, at_nums, header='|Psi|^2 ' + lab)
    Lz = write_lz_cube(os.path.join(out_dir, 'exc_%s_Lz_density.cube' % lab), wf[i],
                       scl, at_pos, at_nums, center=at_pos[-1])
    print("%s : <L_z> about the hole = %+.4f hbar" % (lab, Lz))
