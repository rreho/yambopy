# Yambopy API Documentation

This section contains comprehensive API documentation for all yambopy modules related to optical properties and electron-phonon coupling.

## Package Documentation

### Optical Properties
- [Optical Properties Overview](optical_properties_overview.md) - Complete overview of optical properties modules
- [Base Optical Properties](base_optical_api.md) - Base class for optical calculations
- [Exciton Group Theory](exciton_group_theory_api.md) - Symmetry analysis of exciton states
- [Exciton Dipole](ex_dipole_api.md) - Dipole moment calculations
- [Exciton Phonon](ex_phonon_api.md) - Exciton-phonon coupling
- [Luminescence](luminescence_api.md) - Photoluminescence calculations
- [Point Group Operations](spgrep_point_group_ops_api.md) - Crystallographic symmetry operations
- [Utilities](utils_api.md) - Utility functions

### LetzElPhC Interface
- [LetzElPhC Interface Overview](letzelphc_interface_overview.md) - Complete overview of LetzElPhC interface
- [LetzElPhC Database](lelphcdb_api.md) - Electron-phonon database interface
- [LetzElPhC to Yambo](lelph2y_api.md) - Format conversion utilities

## Class Documentation

### Main Classes
- [BaseOpticalProperties](baseopticalproperties_api.md) - Base class for all optical properties
- [ExcitonGroupTheory](excitongrouptheory_api.md) - Group theory analysis
- [ExcitonDipole](excitondipole_api.md) - Dipole calculations
- [ExcitonPhonon](excitonphonon_api.md) - Exciton-phonon coupling
- [Luminescence](luminescence_api.md) - Luminescence calculations
- [LetzElphElectronPhononDB](letzelphelectronphonondb_api.md) - Electron-phonon database

## Quick Navigation

### By Functionality
- **Symmetry Analysis**: ExcitonGroupTheory, Point Group Operations
- **Optical Properties**: ExcitonDipole, Luminescence
- **Electron-Phonon Coupling**: ExcitonPhonon, LetzElPhC Interface
- **Database Interface**: BaseOpticalProperties, LetzElphElectronPhononDB

### By Usage Level
- **Beginner**: BaseOpticalProperties, ExcitonDipole
- **Intermediate**: ExcitonGroupTheory, Luminescence
- **Advanced**: ExcitonPhonon, Point Group Operations
- **Expert**: LetzElPhC Interface, Custom utilities

## Getting Started

```python
# Import main classes
from yambopy.optical_properties import (
    BaseOpticalProperties, ExcitonGroupTheory, 
    ExcitonDipole, ExcitonPhonon, Luminescence
)
from yambopy.letzelphc_interface import LetzElphElectronPhononDB

# Basic workflow
base = BaseOpticalProperties(path='.')
egt = ExcitonGroupTheory(path='.')
results = egt.analyze_exciton_symmetry(iQ=1, nstates=10)
```

## Notes

- All API documentation is automatically generated from source code docstrings
- Documentation is updated automatically when code changes
- For tutorials and examples, see the tutorials section
- For theoretical background, see the theory section
