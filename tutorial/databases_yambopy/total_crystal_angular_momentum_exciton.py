# Copyright (c) 2025, University of Luxembourg 
# All rights reserved.
#
# Authors: MN
"""
Script to compute total crystal angular momentum
This script loads Yambo databases, calculates the angular momentum for a specific exciton, 
and generates a real-space visualization (.cube file).
"""

import numpy as np
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.dbs import excitondb
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.dbs.wfdb import YamboWFDB
import os

## inputs 
iqpt = 1 
# This is the index of the q-point (momentum transfer). 

path = '.' 
# This defines the working directory where your 'GW_BSE' and 'SAVE' folders are located.
# '.' means the current directory.

gw_bse_dir = 'GW_BSE'
# BSE Job name. This is where your bse file are store after your yambo bse run.

iexe = 1 
# This is the index of the specific exciton state you want to analyze.
# You likely found this index by looking at the absorption spectrum or energy table previously.

degen_tol = 1e-2
# This is the energy tolerance in electron-volts (eV).
# States with energy differences smaller than this value are considered "degenerate" (the same energy)
# and will be treated together during the angular momentum mixing.

# ======= Load data ========

## First load the lattice db
lattice = YamboLatticeDB.from_db_file(os.path.join(path, 'SAVE', 'ns.db1'))
# We load the Lattice Database (ns.db1) from the SAVE folder.
# This file contains information about the crystal structure, unit cell geometry, and k-points.

# load exciton db
filename = 'ndb.BS_diago_Q%d' % (iqpt)
excdb = YamboExcitonDB.from_db_file(lattice, filename=filename,
                                    folder=os.path.join(path, gw_bse_dir), neigs = iexe + 100)
# We load the Exciton Database.
# Critical Note: 'neigs' (number of eigenvalues) is set to 'iexe + 100'. 
# We load more states than the target 'iexe' to ensure we capture all states that might be degenerate 
# with our target. If you don't load the degenerate partners, the physics will be wrong.

# Load the wavefunction database
wfdb = YamboWFDB(path='.', latdb=lattice, bands_range=[np.min(excdb.table[:, 1]) - 1, np.max(excdb.table[:, 2])])

symm_mat_cat = lattice.sym_car[2] 
# The rotation symmetry for which we want to find total_crys_angular_momentum values and simentineous eigenbasis
# For example, here I am selecting 8th symmetry matrix which corresponds to 90 degree rotation along z.
# you can check this in qe scf out file when verbosity = 'high'
# Neverthesless you can also give your custom 3x3 matrix which must satisfy Rq = q and is a symmetry of the crystal.

sbasis = excdb.total_crys_angular_momentum(wfdb, iexe, symm_mat_cat, np.zeros(3), degen_tol=degen_tol)
# This is the core calculation.
# It computes the "Total Crystal Angular Momentum" for the target exciton ('iexe').
# It uses the wavefunctions ('wfdb') and the specific symmetry ('symm_mat_cat').
# 'np.zeros(3)' sets the center of rotation to the origin [0,0,0].
# It returns 'sbasis', which contains the angular momentum values and the new basis vectors.

jvals = sbasis[0]
# These are the angular momentum eigenvalues (the 'j' values) for the degenerate states.

Akcv_j = sbasis[1]
# These are the new coefficients (eigenvectors) for the excitons.
# They represent how to mix the original states to create states with well-defined angular momentum.

print('Total crystal angular momentum : ', jvals)
# Prints the calculated j-values to the screen.

## Plot read space wf
idegen = np.array(excdb.get_degenerate(iexe+1, eps=degen_tol), dtype=int) - 1
# We find the indices of all states that are degenerate with our target exciton.
# 'iexe+1' is used because some internal functions use 1-based indexing, while python uses 0-based.
# The result 'idegen' is an array of Python indices (0-based) for these states.

## Update the existing wfs with the new rotated similentious eigenbasis
excdb.Akcv[idegen] = Akcv_j
# We overwrite the original exciton coefficients in the database object with our new, rotated coefficients.
# This ensures that when we plot the state later, we are plotting the angular momentum eigenstate, 
# not the arbitrary original state.

## we temporartly break degeneracies to plot each state rather than averaging
degen_tol_add = excdb.eigenvalues.max()*4
# We prepare a large number to shift the energies.
# Plotting tools often average degenerate states. To plot a specific 'j' state individually,
# we need to trick the tool into thinking they have different energies.

for i in idegen:
    degen_tol_add += 1 
    # Increment the shift value.
    excdb.eigenvalues[i] += degen_tol_add
    # We artificially shift the energy of this specific state.
    # Now this state is energetically isolated and won't be averaged with others during plotting.

# Now lets plot the real space wavefunction for each j value
print('Computing real space wf for j = %.6f' % (jvals[0]))
# We inform the user which j-value state is being generated.

excdb.real_wf_to_cube(iexe=idegen[0], wfdb=wfdb, fixed_postion=[0.0, 0.0, 0.0], supercell=[9,9,9], degen_tol=1e-14, wfcCutoffRy=-1, fix_particle='e', phase=True)
# This function generates the volumetric data file (.cube).
# 'iexe=idegen[0]': We plot the first state of the degenerate group (corresponding to jvals[0]).
# 'fixed_postion': We fix the position of one particle (usually the hole) at the origin.
# 'fix_particle="e"': This is slightly counter-intuitive naming in some versions, but often implies fixing the electron to see the hole, or vice-versa. 
# 'supercell=[9,9,9]': The plot will span a 9x9x9 grid of unit cells.
# 'degen_tol=1e-14': We use a tiny tolerance here because we manually shifted the energies above; we want it to pick ONLY this one state.
# 'wfcCutoffRy=-1': -1 means we use all G-vectors (no energy cutoff) for maximum precision.

## .cube will be dumped and use vesta to visualize it !
# The script finishes here. You will find a .cube file in your directory which can be opened in VESTA.
