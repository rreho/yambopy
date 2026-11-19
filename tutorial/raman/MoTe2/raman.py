"""
One-phonon excitonic resonant Raman of 1L MoTe2 (Reichardt & Wirtz, Sci. Adv. 2020).

Call chain:
  1. Q=0 BSE (ndb.BS_diago_Q1)          -> exciton energies E_S, eigenvectors A^S
  2. LetzElPhC at q=Gamma (ndb.elph)    -> phonon energies, el-ph matrix elements
  3. exciton_phonon_matelem_iQ(Q=0)     -> G[iq, mode, S_in, S_out] (Ha)
  4. exc_dipoles_pol                    -> D[pol, S] (bohr, photon emission, includes 1/sqrt(Nk))
  5. exc_ph_raman                       -> R[laser, mode, 3, 3]
  6. exc_ph_raman_intensity             -> I[laser, mode]

If a pre-computed `raman_inputs.npz` is present (keys: ph_energies [eV], exc_energies [eV],
exc_dipoles [bohr], exc_ph_mat_el [Ha], cell_vol [bohr^3]) steps 1-4 are skipped.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from yambopy import YamboLatticeDB, YamboExcitonDB, YamboWFDB, LetzElphElectronPhononDB
from yambopy.units import ev2cm1
from yambopy.bse.excitondipoles import exc_dipoles_pol
from yambopy.exciton_phonon.excph_matrix_elements import exciton_phonon_matelem_iQ
from yambopy.exciton_phonon.excph_raman import exc_ph_raman, exc_ph_raman_intensity

path      = '1L'                       # MoTe2 monolayer calculation folder
savepath  = f'{path}/04_nscf/SAVE'     # ns.db1, ns.wf
bsepath   = f'{path}/06_bse/bse'       # ndb.BS_diago_Q1 and ndb.dipoles
elphpath  = f'{path}/07_elph/ndb.elph' # LetzElPhC run with the Gamma point only
npz_file  = 'raman_inputs.npz'

nexc          = -1                     # excitons used (-1: all written by yambo)
mode_window   = [100, 300]             # cm-1: modes kept in the Raman calculation
laser         = np.linspace(0.9, 2.7, 1801) # eV
broad         = 0.1                    # eV (FWHM; Im E_S = broad/2)

if os.path.isfile(npz_file):
    inp = np.load(npz_file)
    ph_energies, exc_energies = inp['ph_energies'], inp['exc_energies']
    exc_dipoles, G, cell_vol = inp['exc_dipoles'], inp['exc_ph_mat_el'], float(inp['cell_vol'])
else:
    lattice = YamboLatticeDB.from_db_file(filename=f'{savepath}/ns.db1')
    cell_vol = abs(np.linalg.det(lattice.lat))
    # 1. Q=0 excitons (Lin = Lout)
    excdb = YamboExcitonDB.from_db_file(lattice, filename='ndb.BS_diago_Q1', folder=bsepath,
                                        Load_WF=True, neigs=nexc)
    exc_energies = excdb.eigenvalues.real
    bands_range = [np.min(excdb.unique_vbands), np.max(excdb.unique_cbands) + 1]
    # 2. phonons and el-ph at Gamma
    elph = LetzElphElectronPhononDB(elphpath, read_all=False)
    iq0 = np.argmin(np.linalg.norm(elph.qpoints, axis=1))
    assert np.allclose(elph.qpoints[iq0], 0.), 'ndb.elph must contain q=Gamma'
    # 3. exciton-phonon matrix elements at Q=q=0 (computed for every q in ndb.elph: use a Gamma-only run)
    wfcs = YamboWFDB(filename='ns.wf', save=savepath, latdb=lattice, bands_range=bands_range)
    G = exciton_phonon_matelem_iQ(elph, wfcs, [excdb], wfcs.Dmat(), Q_in=np.zeros(3), neigs=nexc)
    G, ph_energies = G[iq0], elph.ph_energies[iq0]  # [nmodes, S_in, S_out] (Ha), [nmodes] (eV)
    # 4. exciton dipoles for photon emission (already normalised with 1/sqrt(Nk))
    exc_dipoles = exc_dipoles_pol(savepath, dipoles_path=bsepath, bse_path=bsepath)
    np.savez(npz_file, ph_energies=ph_energies, exc_energies=exc_energies,
             exc_dipoles=exc_dipoles, exc_ph_mat_el=G, cell_vol=cell_vol)

if ph_energies.ndim == 2: ph_energies, G = ph_energies[0], G[0]   # Gamma
freq_cm = ph_energies * ev2cm1
modes = np.where((freq_cm >= mode_window[0]) & (freq_cm <= mode_window[1]))[0]
print('Raman modes (cm-1):', np.round(freq_cm[modes], 1))

# 5. Raman tensor. nkpts=1 because exc_dipoles_pol already contains 1/sqrt(Nk)
R = exc_ph_raman(ph_energies[modes], exc_energies, exc_dipoles, G[modes], laser,
                 broadening=broad, cell_vol=cell_vol, nkpts=1, ph_fre_th=5.0)
np.save('Raman_tensor_MoTe2', R)

# 6. Intensities: unpolarised sum_ij |R_ij|^2 and backscattering parallel XX
I_tot = exc_ph_raman_intensity(R)
I_xx  = exc_ph_raman_intensity(R, e_in=[1, 0, 0])
hdr = 'laser(eV) ' + ' '.join('%.1fcm-1' % f for f in freq_cm[modes])
np.savetxt('Raman_intensities.dat', np.c_[laser, I_tot], header='sum_ij |R_ij|^2: ' + hdr)
np.savetxt('Raman_intensities_xx.dat', np.c_[laser, I_xx], header='|R_xx|^2: ' + hdr)

fig, ax = plt.subplots()
for im, f in enumerate(freq_cm[modes]):
    ax.plot(laser, I_xx[:, im], label='%.1f cm$^{-1}$' % f)
ax.set_xlabel('Laser energy (eV)')
ax.set_ylabel(r'$|R_{xx}|^2$ (a.u.)')
ax.set_yscale('log')
ax.legend()
plt.savefig('MoTe2_raman_excitation_profile.png', dpi=150)
plt.show()
