# One-phonon excitonic resonant Raman: 1L MoTe2

`raman.py` computes the Stokes one-phonon Raman tensor of monolayer MoTe2 as a function
of the laser energy, with excitons from the BSE and exciton-phonon coupling at q = Γ
(Reichardt & Wirtz, Sci. Adv. 6, eabb5915 (2020)).

## Required calculations

| step | code | output used |
|------|------|-------------|
| SCF + NSCF | QE `pw.x` | `04_nscf/SAVE` (`ns.db1`, `ns.wf`) after `p2y` |
| DFPT at Γ | QE `ph.x` | dvscf for LetzElPhC |
| el-ph at Γ | LetzElPhC (q-grid = Γ only) | `07_elph/ndb.elph` |
| BSE at Q = 0 | yambo (TDA, `WRbsWF`, length-gauge dipoles) | `06_bse/bse/ndb.BS_diago_Q1`, `ndb.dipoles` |

Only Γ is needed: `exciton_phonon_matelem_iQ` computes G for every q in `ndb.elph`,
so run LetzElPhC on Γ only.

## Call chain

```python
excdb = YamboExcitonDB.from_db_file(lattice, 'ndb.BS_diago_Q1', folder=bsepath)      # E_S, A^S
elph  = LetzElphElectronPhononDB(elphpath)                                           # Ω_ν, g
wfcs  = YamboWFDB(save=savepath, latdb=lattice, bands_range=bse_bands)
G     = exciton_phonon_matelem_iQ(elph, wfcs, [excdb], wfcs.Dmat(), Q_in=np.zeros(3))[iq_Gamma]
D     = exc_dipoles_pol(savepath, dipoles_path=bsepath, bse_path=bsepath)            # [3, nexc]
R     = exc_ph_raman(ph_energies, exc_energies, D, G, laser, broadening=0.1,
                     cell_vol=cell_vol, nkpts=1)                                     # [nlaser, nmodes, 3, 3]
I_xx  = exc_ph_raman_intensity(R, e_in=[1, 0, 0])                                    # |R_xx|^2
```

Units: energies in eV, G in Ha, dipoles in bohr, volume in bohr³ (same as `exc_ph_luminescence`).
`nkpts=1` because `exc_dipoles_pol` already contains 1/√N_k; pass `nkpts=N_k` with
un-normalised dipoles Σ_kcv A d (PhdScripts convention).

## Formula (code notation)

With ω the laser energy, Ω_m the phonon energy, η = broadening/2, `D = exc_dipoles`,
`G_m[S',S] = exc_ph_mat_el[m, S', S]`:

    R[ω,m,i,j] = sqrt(|ω-Ω_m|/ω) / (nkpts sqrt(V)) Σ_{S,S'} [
        D*_iS G*_m[S',S] D_jS' / ((ω-E_S+iη)(ω-E_S'-Ω_m+iη))
      + D_iS  G_m[S',S]  D*_jS' / ((ω+E_S-iη)(ω+E_S'-Ω_m-iη)) ]

No Bose factor and no ω ω_s³ prefactor are included in R; `exc_ph_raman_intensity(..., ph_temp=T)`
multiplies by n(Ω_m,T)+1.

## Outputs

* `Raman_tensor_MoTe2.npy` — R[laser, mode, 3, 3]
* `Raman_intensities.dat`, `Raman_intensities_xx.dat` — Σ|R_ij|² and |R_xx|² vs laser energy
* `MoTe2_raman_excitation_profile.png`
* `raman_inputs.npz` — cached inputs (reused on the next run)

The port reproduces `compute_Raman_oneph_exc` (PhdScripts, M. Nalabothula) to machine
precision; see `yambopy/exciton_phonon/tests/test_excph_raman.py`.
