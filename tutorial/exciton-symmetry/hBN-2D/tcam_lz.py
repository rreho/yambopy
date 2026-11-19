#
# Exciton symmetry analysis of monolayer hBN at Q = Gamma:
#   irreps -> circular (sigma+/-) basis -> TCAM j (C3) -> hole-centred L_z cubes
#
# Usage:  python tcam_lz.py [calc_dir] [bse_dir] [out_dir] [ewin_eV]
#   calc_dir : folder containing SAVE/            (default '.')
#   bse_dir  : folder with ndb.BS_diago_Q1 and ndb.dipoles, Lkind="bar"
#   out_dir  : where cubes and tcam_summary.npz are written (default '.')
#   ewin_eV  : analyse all multiplets within ewin of the lowest exciton (default 1.0)
#
import os, sys, io, re, contextlib
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
ewin    = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
degen_tol = 1e-2    # eV
supercell = [7, 7, 1]
cutoff_Ry = 40.0
os.makedirs(out_dir, exist_ok=True)

# 0) Load excitons, choose the states: all multiplets within ewin of the lowest one
lat = YamboLatticeDB.from_db_file(os.path.join(path, 'SAVE', 'ns.db1'))
exc = YamboExcitonDB.from_db_file(lat, filename='ndb.BS_diago_Q1', folder=bse_dir)
E = exc.eigenvalues.real
assert np.all(np.diff(E) >= 0), "excitons are expected sorted in energy"
nstates = int(np.sum(E < E[0] + ewin))
while nstates < len(E) and E[nstates] - E[nstates - 1] < degen_tol: nstates += 1

# 1) Irreducible representations (D3h) of the exciton multiplets
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    compute_exc_rep(path=path, bse_dir=bse_dir, iqpt=1, nstates=nstates,
                    degen_tol=degen_tol, symm_tol=0.06)
print(buf.getvalue())
irreps = re.findall(r'^\s*([\d.]+)\s+(\d+)\s+:\s+(.+?)\s*$', buf.getvalue(), re.M)

# 2) Circular basis inside each degenerate bright block (emission dipoles)
dip = exc_dipoles_pol(os.path.join(path, 'SAVE'), dipoles_path=bse_dir,
                      bse_path=bse_dir, save_files=False)        # (3, nexc)
U, blocks, dip = rotate_excdb_to_circular(exc, dip, atol=degen_tol, verbose=True)
dip = dip[:, :nstates]
sp = np.abs(dip.T @ np.conj(E_PLUS))**2
sm = np.abs(dip.T @ np.conj(E_MINUS))**2
s2 = (np.abs(dip)**2).sum(0)
splus = np.where(sp + sm > 1e-8 * s2.max(), sp / np.maximum(sp + sm, 1e-300), np.nan)

# 3) TCAM from the C3 (counter-clockwise, about z) representation matrix
wfdb = YamboWFDB(path=path, latdb=lat,
                 bands_range=[np.min(exc.table[:, 1]) - 1, np.max(exc.table[:, 2])])
symm = Crystal_Symmetries(lat, tol=1e-4)
ic3 = find_rotation(symm.rotations, 3)
R3, t3 = symm.rotations[ic3], symm.translations[ic3]
dmats = wfdb.Dmat(symm_mat=R3[None], frac_vec=t3[None], time_rev=False)[0]
Ak = exc.get_Akcv()[:nstates].copy()
D3 = exciton_rotation_rep(Ak, R3, wfdb, dmats, tau_cart=t3)
c3diag = np.diag(D3).copy()
j_state = -3 * np.angle(c3diag) / (2 * np.pi)
dark_blocks = []
print("\nTCAM per multiplet (eigenvalues of D(C3) in the block):")
for blk in find_degeneracy_evs(E[:nstates], atol=degen_tol):
    blk = np.sort(blk)
    j, lam, V = tcam_from_rotation(D3[np.ix_(blk, blk)], 3)
    print("  states %-16s E=%.4f eV  j = %s" % ((blk + 1).tolist(), E[blk].mean(),
                                                 np.round(j, 3).tolist()))
    if s2[blk].max() < 1e-2 * s2.max():
        dark_blocks.append(blk)
        # dark multiplet: rotate to the C3 (TCAM) eigenbasis |j> = sum_m V[m,j] |m>
        # (bright blocks are already C3 eigenstates after the circular rotation)
        if np.any(np.abs(np.abs(c3diag[blk]) - 1) > 1e-2):
            Ak[blk] = np.tensordot(V.T, Ak[blk], axes=(1, 0))
            dip[:, blk] = dip[:, blk] @ V
            c3diag[blk], j_state[blk] = lam, j
sp = np.abs(dip.T @ np.conj(E_PLUS))**2
sm = np.abs(dip.T @ np.conj(E_MINUS))**2
s2 = (np.abs(dip)**2).sum(0)
splus = np.where(sp + sm > 1e-8 * s2.max(), sp / np.maximum(sp + sm, 1e-300), np.nan)
j_state[np.abs(np.abs(c3diag) - 1) > 1e-2] = np.nan   # not a C3 eigenstate
print("\n  S    E (eV)   |D|^2/max  sigma+ frac  <C3> (|.|, deg)   j")
for n in range(nstates):
    print(" %3d  %8.4f  %8.4f   %8.3f    (%.3f, %7.1f)  %+5.2f"
          % (n + 1, E[n], s2[n] / s2.max(), splus[n], abs(c3diag[n]),
             np.degrees(np.angle(c3diag[n])), j_state[n]))

# 4) Real space (hole 1 bohr above N): lowest sigma+/- doublet and lowest dark pair
bright = [n for n in range(nstates) if s2[n] > 1e-2 * s2.max()][:2]
dark = [n for n in dark_blocks[0] if abs(j_state[n]) < 0.5][:2] if dark_blocks else []
sel = dark + bright
iN = int(np.where(lat.atomic_numbers == 7)[0][0])
hole = np.linalg.solve(lat.lat.T, lat.car_atomic_positions[iN]) % 1
hole[2] += 1.0 / lat.lat[2, 2]
scl, at_nums, at_pos, wf = ex_wf2Real(Ak[sel], np.zeros(3), wfdb,
                                      [np.min(exc.table[:, 1]), np.max(exc.table[:, 2])],
                                      hole, fix_particle='h', supercell=supercell,
                                      wfcCutoffRy=cutoff_Ry)
Lz_sel = []
for i, n in enumerate(sel):
    kind = ('dark_j%+d' % round(j_state[n])) if n in dark else ('sigma_plus' if sp[n] > sm[n] else 'sigma_minus')
    lab = 'S%d_%s_%.4feV' % (n + 1, kind, E[n])
    psi = wf[i].reshape((-1,) + wf.shape[-3:])          # spinor channels
    dens = (np.abs(psi)**2).sum(0)
    ch = np.argmax((np.abs(psi)**2).sum(axis=(1, 2, 3)))   # dominant channel (gauge-dependent Re/Im)
    nrm = np.abs(psi[ch]).max()
    for tag, dat in (('density', dens / dens.max()), ('Re', psi[ch].real / nrm),
                     ('Im', psi[ch].imag / nrm)):
        write_cube(os.path.join(out_dir, 'exc_%s_%s.cube' % (lab, tag)), dat, scl,
                   at_pos, at_nums, header='%s %s (spinor channel %d for Re/Im)' % (tag, lab, ch))
    Lz = write_lz_cube(os.path.join(out_dir, 'exc_%s_Lz_density.cube' % lab), wf[i],
                       scl, at_pos, at_nums, center=at_pos[-1])
    Lz_sel.append(Lz)
    print("%s : <L_z> about the hole = %+.4f hbar" % (lab, Lz))

np.savez(os.path.join(out_dir, 'tcam_summary.npz'), energies=E[:nstates],
         irrep_energies=np.array([float(r[0]) for r in irreps]),
         irrep_degeneracy=np.array([int(r[1]) for r in irreps]),
         irrep_labels=np.array([r[2] for r in irreps]),
         j=j_state, c3_diag=c3diag, sigma_plus_fraction=splus,
         dipole_strength=s2 / s2.max(), cube_states=np.array(sel) + 1,
         Lz_cube_states=np.array(Lz_sel), hole_cart=at_pos[-1])
