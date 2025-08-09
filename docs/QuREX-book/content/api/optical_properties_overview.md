# Optical Properties Package API Reference

This page provides an overview of all modules and classes in the `yambopy.optical_properties` package.

## Package Overview

The `yambopy.optical_properties` package contains the following modules:

- **`base_optical`**: Base class for optical properties calculations
- **`exciton_group_theory`**: Group theory analysis of exciton states
- **`ex_dipole`**: Exciton dipole moment calculations
- **`ex_phonon`**: Exciton-phonon coupling analysis
- **`luminescence`**: Luminescence and photoluminescence calculations
- **`spgrep_point_group_ops`**: Point group operations using spgrep library
- **`utils`**: Utility functions for optical properties calculations

## Quick Start

```python
# Import the entire package
import yambopy.optical_properties

# Or import specific modules
from yambopy.optical_properties.base_optical import *
from yambopy.optical_properties.exciton_group_theory import *
from yambopy.optical_properties.ex_dipole import *
from yambopy.optical_properties.ex_phonon import *
from yambopy.optical_properties.luminescence import *
from yambopy.optical_properties.spgrep_point_group_ops import *
from yambopy.optical_properties.utils import *

```

## Module Documentation

- [Base Optical](base_optical_api.md)
- [Exciton Group Theory](exciton_group_theory_api.md)
- [Ex Dipole](ex_dipole_api.md)
- [Ex Phonon](ex_phonon_api.md)
- [Luminescence](luminescence_api.md)
- [Spgrep Point Group Ops](spgrep_point_group_ops_api.md)
- [Utils](utils_api.md)

## Common Usage Patterns

### Optical Properties Analysis

```python
from yambopy.optical_properties.base_optical import BaseOpticalProperties
from yambopy.optical_properties.exciton_group_theory import ExcitonGroupTheory
from yambopy.optical_properties.ex_dipole import ExcitonDipole

# Group theory analysis
egt = ExcitonGroupTheory(path='.')
results = egt.analyze_exciton_symmetry(iQ=1, nstates=10)

# Dipole analysis
dipole = ExcitonDipole(path='.')
dipole_results = dipole.compute()
```

### LetzElPhC Interface

```python
from yambopy.letzelphc_interface.lelphcdb import LetzElphElectronPhononDB

# Read electron-phonon database
lelph_db = LetzElphElectronPhononDB('path/to/ndb.elph')
print(f"Number of k-points: {lelph_db.nk}")
print(f"Number of q-points: {lelph_db.nq}")
```

## Notes

- All API documentation is automatically generated from source code docstrings
- For theoretical background, see the theory documentation
- For practical examples, see the tutorials and example notebooks
