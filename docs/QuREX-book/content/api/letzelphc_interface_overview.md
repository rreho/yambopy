# Letzelphc Interface Package API Reference

This page provides an overview of all modules and classes in the `yambopy.letzelphc_interface` package.

## Package Overview

The `yambopy.letzelphc_interface` package contains the following modules:

- **`lelphcdb`**: Interface to LetzElPhC electron-phonon databases
- **`lelph2y`**: Conversion utilities between LetzElPhC and Yambo formats

## Quick Start

```python
# Import the entire package
import yambopy.letzelphc_interface

# Or import specific modules
from yambopy.letzelphc_interface.lelphcdb import *
from yambopy.letzelphc_interface.lelph2y import *

```

## Module Documentation

- [Lelphcdb](lelphcdb_api.md)
- [Lelph2Y](lelph2y_api.md)

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
