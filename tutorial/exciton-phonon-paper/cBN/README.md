# Cubic BN: phonon-assisted luminescence (paper figure `fig_cbn`)

These files reproduce the cBN figure of the yambopy/LetzElPhC methods paper:
- (a) phonon dispersion with the four phonons at X;
- (b) the 10 K spectrum resolved by phonon branch;
- (c) replica ratios versus the number of intermediate excitons.

## Quick start: figure from the processed data
```
cd plot && python fig_cbn.py         # -> ../figures/fig_cbn.pdf/.png
```
This needs numpy, matplotlib and SciencePlots. The style is set in `plot/paperstyle.py`.

## Full calculation
LDA, SG15 norm-conserving pseudopotentials (`pseudo/`), a = 3.581 Å (relaxed), 80 Ry, 16³ grid for k, Q and q.

| step | input | code | notes |
|---|---|---|---|
| 1 | `inputs/02_dfpt/scf.in`, `ph.in` | `pw.x`, `ph.x`, then `lelphc -pp --code=qe -F ph.in` | scf on 8³; DFPT on a 16³ q grid (145 irreducible q); `dvscf` written for LetzElPhC |
| 2 | `inputs/01_nscf/nscf.in` | `pw.x`, then `p2y` and `yambo` (init) | 16³ k grid, 100 bands → `SAVE` |
| 3 | `inputs/03_gw/gw.in` | `yambo -F gw.in -J GW` | G₀W₀ PPA; quasiparticle energies for bands 2–8 at all 145 k |
| 4 | `inputs/04_bse/bser_*.in` | `yambo -F bser_A_B.in -J "BSEr_A_B,GW"` | Lkind full, bands 2–8, Q = 1–145 in six chunks, 12 states per Q (SLEPc); per-k GW corrections `KfnQPdb "E < ./GW/ndb.QP"` |
| 4b | `inputs/04_bse/bse_bar600.in` | `yambo -F bse_bar600.in -J "BSEr_bar600,GW"` | Q = 0 without long-range exchange (Lkind bar), 600 states. The 200- and 400-state runs differ only in `BSSNEig`. |
| 5 | `inputs/05_elph/lelphc.in` | `lelphc -F lelphc.in` | electron–phonon matrix elements, bands 2–8, on the nscf SAVE |
| 6 | `inputs/06_phonon_dispersion/q2r.in`, `matdyn.in` | `q2r.x`, `matdyn.x` | dispersion along L–Γ–X–W–K, including ε∞ and Born charges (LO–TO splitting) |

The drivers in `drivers/` expect this layout, relative to the directory they are run from:
```
runs/01_nscf/SAVE          # yambo SAVE of step 2
runs/03_gw/GW              # ndb.QP and ndb.dipoles of step 3
runs/04_bse/BSE_all        # symlinks to every ndb.BS_diago_Q* of the six chunks of step 4
runs/04_bse/BSE_bar200|400|600   # Q=0 Lbar runs of step 4b
runs/05_elph               # ndb.elph of step 5
data/                      # outputs
```

The PL drivers use the yambopy paper branch: `exc_ph_get_inputs` builds and caches the exciton–phonon matrix elements, then `exc_ph_luminescence` computes the spectrum.

| driver | config | output (in `data/`) |
|---|---|---|
| `drivers/pl_nexc/pl_nexc.py` | `config.json`: n_in = 12, 50, 100, 200 | `cBN_PL_nexc_conv_16k.npz` |
| `drivers/pl_nexc400/pl_nexc.py` | n_in = 200, 300, 400 | `cBN_PL_nexc400_conv_16k.npz` |
| `drivers/pl_nexc600/pl_nexc.py` | n_in = 400, 500, 600 | `cBN_PL_nexc600_conv_16k.npz` |
| `drivers/pl_nexc600/pl_branch600.py` | same config; phonon-branch decomposition (TA, LA, TO, LO) at n_in = 600 | `cBN_PL_branches600_16k.npz` |

Run each driver as `python <driver> config.json`.

The branch decomposition labels each mode by its longitudinal fraction relative to q. It reproduces the total spectrum of `exc_ph_luminescence`, with the same prefactor and broadening.

## Parameters of the spectra
- 10 and 300 K.
- 12 emitting excitons per Q.
- 5 meV Lorentzian broadening (FWHM).
- Exciton and phonon temperatures equal.
- Both phonon channels (emission and absorption).
