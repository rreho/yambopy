import numpy as np

from qepy.lattice import Path
from yambopy.bse.bse_dispersion import ExcitonDispersion
from yambopy.dbs.latticedb import YamboLatticeDB

path = './'
npoints = 30

symm_path = Path([[[0.0, -0.5, 0.0], 'M'],
                  [[0.0,  0.0, 0.0], '$\Gamma$'],
                  [[1/3, 1/3,  0.0], 'K'],
                  [[2/3, -1/3, 0.0], 'Kp'],
                  [[-2/3, 1/3, 0.0], 'Kpp'],
                  [[0.0, -0.5, 0.0], 'M'],
                  [[0.0,  0.0, 0.0], '$\Gamma$']],
                 [npoints*3, npoints*3, npoints*3])

symm_path.get_klist()

latticedb = YamboLatticeDB.from_db_file(filename=path+'SAVE/ns.db1')

nexcitons = 10
size_dot  = 80
ymax      = 6.1

# load_eigenvectors=True required for orbital projection
excdisp = ExcitonDispersion(lattice=latticedb, nexcitons=nexcitons,
                            folder=path+"BSE", load_eigenvectors=True)
print(excdisp)

# ------------------------------------------------------------------
# 1. Plain dispersion (no spin, no interpolation)
# ------------------------------------------------------------------
fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3, ylim=(3.8, ymax)
)
fig.savefig("short/exciton_dispersion.png")
print("[SAVED] exciton_dispersion.png")

# ------------------------------------------------------------------
# 2. Interpolated dispersion
# ------------------------------------------------------------------
fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(3.8, ymax), expand_bz=True,
    interpolate=True, method='cubic_spline',
)
fig.savefig("short/interpolated_exciton_dispersion.png")
print("[SAVED] interpolated_exciton_dispersion.png")

# ------------------------------------------------------------------
# 3. Spin-projected dispersion (rotate_Ak; Rzz method removed)
# ------------------------------------------------------------------
spin_data = excdisp.get_spin_along_path(
    symm_path, expand_bz=True, tol=1e-3,
    save_dir='SAVE', bse_dir='BSE',
    contribution='b', dmat_mode='run', dmat_file='Dmats.npy',
)

fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(3.8, ymax), expand_bz=True, spin_data=spin_data,
)
fig.savefig("short/exciton_spin_dispersion.png")
print("[SAVED] exciton_spin_dispersion.png")

fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(3.8, ymax), expand_bz=True, spin_data=spin_data,
    interpolate=True, method='cubic_spline',
)
fig.savefig("short/interpolated_exciton_spin_dispersion.png")
print("[SAVED] interpolated_exciton_spin_dispersion.png")

# ------------------------------------------------------------------
# 4. Orbital-projected dispersion
#    Requires projwfc rerun on the 19 IBZ k-points.
#    Run: pw.x -i hBN-2D.nscf > log_nscf_ibz.out  (in bands/)
#         projwfc.x -i proj.in > projwfc.log        (in bands/)
# ------------------------------------------------------------------
from qepy.projwfcxml import ProjwfcXML

proj = ProjwfcXML(
    'hBN_2D',
    output_filename='projwfc.log',
    path='/Users/riccardo.reho/workQE/Projects/hBN-2D/bands',
)

B_p = proj.get_states_helper(['B'], ['p'])   # B pz, px, py
N_p = proj.get_states_helper(['N'], ['p'])   # N pz, px, py

orbital_groups = [
    {'orbitals': B_p, 'color': 'royalblue', 'label': 'B-p'},
    {'orbitals': N_p, 'color': 'tomato',    'label': 'N-p'},
]

fig, ax = excdisp.plot_orbital_projected_dispersion(
    symm_path, proj, orbital_groups,
    contribution='both', expand_bz=True, tol=1e-3,
    s=2000, ylim=(3.8, ymax), figsize=(16, 8),
)
fig.savefig("short/orbital_projected_exciton_dispersion.png")
print("[SAVED] orbital_projected_exciton_dispersion.png")

fig, ax = excdisp.plot_orbital_projected_dispersion(
    symm_path, proj, orbital_groups,
    contribution='both', expand_bz=True, tol=1e-3,
    interpolate=True, method='cubic_spline',
    s=2000, lw=1.5, line_color='black',
    ylim=(3.8, ymax), figsize=(16, 8),
)
fig.savefig("short/orbital_projected_exciton_dispersion_interp.png")
print("[SAVED] orbital_projected_exciton_dispersion_interp.png")
