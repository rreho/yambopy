import numpy as np

from qepy.lattice import Path
from yambopy.bse.bse_dispersion import ExcitonDispersion
from yambopy.dbs.latticedb import YamboLatticeDB

path = './'
npoints = 30
# 0.000000000 |  0.000000000 |  0.000000000 |   # Gamma
# 0.0000000000|  0.500000000  | 0.0000000000|  # M
# 0.333333333 |  0.333333333  |  0.000000000|  # K
# 0.0000000000|  0.000000000  | 0.0000000000|  # Gamma
# -0.333333333|  0.666666666  | 0.000000000    $ K'
# -0.500000000 |  0.500000000 |  0.00000000|   # M'
symm_path = Path([[[0.0,  0.0, 0.0], '$\Gamma$'],
                  [[0.0, 0.5, 0.0], 'M'],
                  [[1/3, 1/3,  0.0], 'K'],
                  [[-0.5,  0.5, 0.0], 'Mp'],
                  [[-1/3, 2/3, 0.0], 'Kp'],
                  #[[-2/3, 1/3, 0.0], 'Kpp'],
                  #[[0.0, -0.5, 0.0], 'M'],
                  [[0.0,  0.0, 0.0], '$\Gamma$']],
                 [npoints,npoints,npoints,npoints,npoints])

symm_path.get_klist()

latticedb = YamboLatticeDB.from_db_file(filename=path+'SAVE/ns.db1')

nexcitons = 8
size_dot  = 80
ymax      = 2.2
ymin      = 1.7
# run_eigenvectors=True required for orbital projection
excdisp = ExcitonDispersion(lattice=latticedb, nexcitons=nexcitons,
                            folder=path+"GW_BSE_full", load_eigenvectors=True)
excdisp_tilde = ExcitonDispersion(lattice=latticedb,nexcitons=nexcitons, folder=path+"GW_BSE_tilde", load_eigenvectors=True)
print(excdisp)
# ------------------------------------------------------------------
# 1. Plain dispersion (no spin, no interpolation)
# ------------------------------------------------------------------
fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3, ylim=(3.8, ymax)
)
fig.savefig("exc_bands/exciton_dispersion.png")
print("[SAVED] exciton_dispersion.png")

# ------------------------------------------------------------------
# 2. Interpolated dispersion
# ------------------------------------------------------------------
fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(ymin, ymax), expand_bz=True,
    interpolate=True, method='cubic_spline',
)
fig.savefig("exc_bands/interpolated_exciton_dispersion.png")
print("[SAVED] interpolated_exciton_dispersion.png")

fig, ax = excdisp_tilde.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(ymin, ymax), expand_bz=True,
    interpolate=True, method='cubic_spline',
)
fig.savefig("exc_bands/interpolated_exciton_dispersion_tilde.png")
print("[SAVED] interpolated_exciton_dispersion.png")
# ------------------------------------------------------------------
# 3. Spin-projected dispersion (rotate_Ak; Rzz method removed)
# ------------------------------------------------------------------

spin_data =  excdisp.get_spin_along_path(symm_path,expand_bz=True, tol=1e-3, save_dir='SAVE', bse_dir='GW_BSE_full',
                            contribution='b',
                            dmat_mode='run', dmat_file='Dmats.npy')

spin_data_tilde =  excdisp.get_spin_along_path(symm_path,expand_bz=True, tol=1e-3, save_dir='SAVE', bse_dir='GW_BSE_tilde',
                            contribution='b',
                            dmat_mode='run', dmat_file='Dmats.npy')

fig, ax = excdisp.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(ymin, ymax), expand_bz=True, spin_data=spin_data,
    interpolate=True, method='cubic_spline',
)
fig.savefig("exc_bands/interpolated_exciton_spin_dispersion.png")
print("[SAVED] interpolated_exciton_spin_dispersion.png")

fig, ax = excdisp_tilde.plot_exciton_dispersion(
    symm_path, figsize=(16, 8), s=size_dot, lpratio=3,
    ylim=(ymin, ymax), expand_bz=True, spin_data=spin_data_tilde,
    interpolate=True, method='cubic_spline',
)
fig.savefig("exc_bands/interpolated_exciton_spin_dispersion_tilde.png")
# ------------------------------------------------------------------
# 4. Orbital-projected dispersion
#    Requires projwfc rerun on the 19 IBZ k-points.
#    Run: pw.x -i hBN-2D.nscf > log_nscf_ibz.out  (in bands/)
#         projwfc.x -i proj.in > projwfc.log        (in bands/)
# ------------------------------------------------------------------
from qepy.projwfcxml import ProjwfcXML

proj = ProjwfcXML(
    'mos2',
    output_filename='projwfc.log',
    path='/mnt/scratch/users/rreho/MoS2_MN/bands_48x48x1',
)
Mo_d = proj.get_states_helper(['Mo'], ['d'])   # B pz, px, py
S_p = proj.get_states_helper(['S'], ['p'])   # N pz, px, py

orbital_groups = [
    {'orbitals': Mo_d, 'color': 'royalblue', 'label': 'Mo-d'},
    {'orbitals': S_p, 'color': 'tomato',    'label': 'S-p'},
]

fig, ax = excdisp.plot_orbital_projected_dispersion(
    symm_path, proj, orbital_groups,
    contribution='both', expand_bz=True, tol=1e-3,
    interpolate=True, method='cubic_spline',
    s=500, lw=1.5, line_color='black',
    ylim=(ymin, ymax), figsize=(16, 8),
)
fig.savefig("excdisp_tilde/orbital_projected_exciton_dispersion_interp.png")
print("[SAVED] orbital_projected_exciton_dispersion_interp.png")

fig, ax = excdisp.plot_orbital_projected_dispersion(
    symm_path, proj, orbital_groups,
    contribution='both', expand_bz=True, tol=1e-3,
    interpolate=True, method='cubic_spline',
    s=500, lw=1.5, line_color='black',
    ylim=(ymin, ymax), figsize=(16, 8),
)
fig.savefig("excdisp_tilde/orbital_projected_exciton_dispersion_interp_tilde.png")
print("[SAVED] orbital_projected_exciton_dispersion_interp_tilde.png")
