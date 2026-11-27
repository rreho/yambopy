# Tutorials for the methods paper (yambopy + LetzElPhC)

Each folder reproduces one figure or one set of results of the paper. Each folder has:
- `inputs/`: the Quantum ESPRESSO, Yambo and LetzElPhC inputs of every step;
- `drivers/`: the yambopy scripts that turn the databases into the processed data;
- `data/`: the processed data (small .npz files);
- `plot/`: the figure script;
- a README with the run order.

The raw databases (wavefunctions, BSE and electron–phonon databases) are too large for the repository. Rerun them from `inputs/`, or download them from the data archive of the paper.

| folder | paper | status |
|---|---|---|
| `cBN/` | cubic BN: phonons at X, branch-resolved spectrum, convergence in intermediate excitons | final |
| `diamond/` | diamond: mechanism (electron/hole channels), state-resolved numerator, spectrum vs experiment | to be added after the 16³ reference run |
| `BN-layered/` | hBN (AA′) and Bernal BN vs Rousseau et al., PRB 106, 035203 (2022) | calculations running |
| `phosphorene/` | resonant Raman of monolayer black phosphorus | calculations running |

Related tutorials on this branch:
- exciton symmetry and TCAM (`../exciton-symmetry/hBN-2D`);
- exciton dispersion (`../exciton-dispersion`);
- resonant Raman of MoTe₂ (`../raman/MoTe2`);
- exciton–phonon basics (`../exciton-phonon`).

Requirements: yambopy from this branch, numpy, scipy, matplotlib, netCDF4, SciencePlots. The full calculations use Quantum ESPRESSO 7.5, Yambo 5.4 and LetzElPhC.
