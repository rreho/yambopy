# Wannier-Yambo Interface

This module provides an interface between Yambo, Wannier90, and external codes for computing screening matrix elements in the Wannier basis.

## Overview

The interface implements the computation of:

1. **Density matrices** $\rho_{\mathbf{q}}^{n \mathbf{R}}(\mathbf{r})$:
   ```
   ρ_q^nR(r) = (1/Nk) exp(-iq·R) Σ_k ũ*_nk(r) ũ_n,k+q(r)
   ```
   where ũ_nk are Wannier-rotated wavefunctions:
   ```
   ũ_nk(r) = Σ_m U_mn(k) u_mk(r)
   ```

2. **Bare potentials** $V_{nm}(\mathbf{R})$ (TODO):
   ```
   V_q^nR(r) = ∫ d³r' v(r,r') ρ_q^nR(r')
   V_nm(R) = Σ_q ∫ d³r ρ*_q^nR(r) V_q^m0(r)
   ```

3. **Screened potentials** $W_{nm}(\mathbf{R})$ (TODO):
   ```
   Δρ_q^n0(r) = ∫ d³r' χ_q(r,r') V_q^n0(r')
   W_nm(R) = V_nm(R) + Σ_q ∫ d³r V*_q^nR(r) Δρ_q^n0(r)
   ```

## Current Implementation Status

### ✅ Implemented

- [x] Loading Yambo wavefunctions (via YamboWFDB)
- [x] Loading Wannier90 U matrices from .amn file
- [x] Loading R vectors from _hr.dat file
- [x] Wannier rotation of wavefunctions
- [x] k+q point finding with periodic boundary conditions
- [x] Basic density matrix computation ρ_q^nR in G-space
- [x] IBZ to BZ k-point mapping

### 🚧 Work in Progress / TODO

- [ ] Proper G-vector matching between k and k+q
- [ ] Full BZ summation (currently only IBZ with weights)
- [ ] Symmetry operations on wavefunctions
- [ ] Reading U matrix from .chk file
- [ ] Reading U matrix from UNK*.NC files
- [ ] Real-space representation (FFT conversion)
- [ ] Loading screening database (em1s)
- [ ] Bare potential computation V_nm(R)
- [ ] Screened potential computation W_nm(R)
- [ ] q-point grid handling for potential integrals
- [ ] Parallelization of k-point loops
- [ ] Memory optimization for large systems
- [ ] Comprehensive tests

## Usage Example

```python
from yambopy.wannier import WannierYamboInterface

# Initialize the interface
interface = WannierYamboInterface(
    save_path="path/to/SAVE",
    wannier_path="path/to/wannier90",
    seedname="mos2",
    bands_range=[0, 26]  # Load bands 0-25
)

# Load U matrix and R vectors
interface.load_U_matrix_from_amn()
interface.load_R_vectors_from_hr()

# Compute density matrix for q=(0,0,0) and all R
q_vec = np.array([0.0, 0.0, 0.0])
rho_all = interface.compute_all_rho_q_R(
    q_vec=q_vec,
    ispin=0,
    return_gspace=True
)

# rho_all has shape [nR, nwann, ngvecs]
print(f"Computed rho with shape: {rho_all.shape}")
```

See `examples/wannier_yambo_interface_example.py` for a complete working example.

## File Structure

```
yambopy/wannier/
├── wann_yambo_interface.py   # Main interface class
├── wann_io.py                 # Wannier90 file readers (AMN, HR, etc.)
├── README_interface.md        # This file
└── ...                        # Other Wannier-related modules

examples/
└── wannier_yambo_interface_example.py  # Usage examples
```

## Technical Notes

### K-point Handling

The implementation uses a KDTree for efficient k-point searching with periodic boundary conditions. The workflow is:

1. Load k-points in IBZ from Yambo
2. Expand to full BZ using symmetries
3. For k+q finding:
   - Add q to k in crystal coordinates
   - Use KDTree to find k+q in BZ (handles wrapping)
   - Map back to IBZ using symmetry information

### G-space vs Real-space

Currently, calculations are done in G-space:
- Wavefunctions: u_nk(G)
- Density matrices: ρ_q^nR(G)
- Potentials: V_q(G), W_q(G)

Real-space conversion via FFT is planned but not fully implemented.

### Symmetries

The code currently uses symmetry information from Yambo to map BZ k-points to IBZ. However, symmetry operations on the wavefunctions themselves (rotation matrices, phase factors) are not yet fully implemented.

### Memory Considerations

For large systems:
- Wavefunctions can be large: [nk, nspin, nbands, nspinor, ngvecs]
- Consider loading bands in chunks
- Use `bands_range` parameter to limit memory usage

## Dependencies

- numpy
- netCDF4
- yambopy.dbs.wfdb (YamboWFDB)
- yambopy.dbs.em1sdb (YamboStaticScreeningDB)
- yambopy.kpoints (k-point utilities)
- yambopy.wannier.wann_io (Wannier90 readers)

## References

1. Wannier90: http://www.wannier.org/
2. Yambo: http://www.yambo-code.org/
3. Wannier interpolation: Marzari & Vanderbilt, PRB 56, 12847 (1997)

## Contributing

This is an active development. Key areas for contribution:

1. **Testing**: Need comprehensive tests with known results
2. **Optimization**: K-point loops can be parallelized
3. **G-vector matching**: Improve the convolution between k and k+q
4. **Documentation**: Add more examples and use cases
5. **Validation**: Compare with direct DFT calculations

## Contact

For questions or issues, please contact the yambopy developers or open an issue on GitHub.