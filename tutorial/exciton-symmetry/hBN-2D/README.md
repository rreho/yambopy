# Exciton symmetry, TCAM and L_z density: monolayer hBN

`tcam_lz.py` runs the whole symmetry analysis of the Q = Γ excitons of monolayer hBN:

1. **Irreps.** `compute_exc_rep` builds the exciton representation matrices of the D3h
   little group and splits each multiplet into irreps.
2. **Circular basis.** `exc_dipoles_pol` gives the emission dipoles. `rotate_excdb_to_circular`
   rotates each degenerate bright block into pure σ+ / σ− states.
3. **TCAM.** `find_rotation` picks the counter-clockwise C3 about z. `exciton_rotation_rep`
   builds D(C3) on the exciton states, and `tcam_from_rotation` diagonalises each multiplet:
   D(C3) = exp(-2πi J/3), λ = e^{-2πi j/3}, with j taken in (-3/2, 3/2].
4. **L_z cubes.** `ex_wf2Real` computes Ψ(r_e; r_h) with the hole 1 bohr above N.
   `write_lz_cube` writes the gauge-invariant density l_z = Im[Ψ*(x∂y − y∂x)Ψ] about the hole,
   together with <L_z> (in units of ħ).

## Requirements
- `SAVE/` from a non-collinear or collinear nscf run, converted with `p2y`.
- A BSE run at Q = Γ with **`Lkind = "bar"`**. `"full"` breaks the Γ symmetry along BLongDir.
  The run must also have `WRbsWF`, so that `ndb.BS_diago_Q1` holds the eigenvectors,
  and its `ndb.dipoles`.
- Enough eigenvalues (`BSSNEig`) to close every multiplet you analyse.

```
python tcam_lz.py <calc_dir_with_SAVE> <bse_dir> <out_dir> [ewin_eV=1.0]
```

## Expected output (12×12×1, no SOC, BSEBands 2–5, BSSmod = "d")
```
 4.1016 eV  E'   states [1,2]  sigma+ / sigma-  j = +1 / -1
 5.0390 eV  E'   states [3,4]                     j = +1 / -1
 5.1709 eV  A2'                                   j =  0
 5.2106 eV  A1'                                   j =  0
S1 sigma+ : <L_z> about the hole = +1.54 hbar   (l_z > 0, counter-clockwise)
S2 sigma- : <L_z> about the hole = -1.54 hbar
```
σ+ goes with j = +1, because e₊ = (x + iy)/√2 transforms with λ = e^{-2πi/3}.
On the real-space grid this gives Ψ(C3 r) = e^{+2πi/3} Ψ(r) about the hole axis.

Outputs: density/Re/Im/L_z cubes for the lowest sigma+/- doublet and the lowest j=0 pair of the
lowest dark multiplet (dark multiplets are first rotated to the C3 eigenbasis), plus `tcam_summary.npz`
(energies, irreps, j, <C3>, sigma+ fraction, dipole strength, <L_z> of the cube states).

## Notes
- Re/Im cubes of Ψ depend on the gauge (the global phase of each eigenvector), so they cannot
  show the winding. Use the `*_Lz_density.cube` files: plot isosurfaces at ±0.2 in VESTA.
  Each chirality shows a single sign.
- Symmetry only protects the C3 eigenvalue mod 3. <L_z> is not quantised: it depends on where
  the hole is placed (+1.54 with the hole on N, +1.19 with the hole on B).
- With spinors (`Dmat` includes SU(2)), j is the total angular momentum, orbital plus spin.
  l_z is orbital only and is summed over the spinor channels.
