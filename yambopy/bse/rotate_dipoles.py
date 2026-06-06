"""
Rotate electronic dipoles under symmetry operations.
Similar to rotate_excitonwf.py but for transition dipole moments.
"""
import numpy as np

def rotate_dipole_vector(d_cvk, rot_mat, time_rev=False):
    """
    Rotate dipole vector under symmetry operation.

    Dipoles are vectors: d'_i = R_ij * d_j
    Under time-reversal: d → d*

    Parameters
    ----------
    d_cvk : ndarray
        Electronic dipole with shape (..., 3) for Cartesian components
    rot_mat : ndarray
        3x3 rotation matrix in Cartesian coordinates
    time_rev : bool
        Apply time-reversal symmetry (conjugation)

    Returns
    -------
    ndarray
        Rotated dipole
    """
    # Rotate Cartesian components
    if d_cvk.ndim == 1:
        # Single vector
        d_rotated = rot_mat @ d_cvk
    else:
        # Multiple vectors (e.g., multiple bands)
        # Assume last axis is Cartesian
        d_rotated = np.einsum('ij,...j->...i', rot_mat, d_cvk)

    # Apply time-reversal
    if time_rev:
        d_rotated = np.conj(d_rotated)

    return d_rotated

def rotate_dipoles_ibz_to_fbz(dipoles_ibz, latdb, time_rev_list=None):
    """
    Expand electronic dipoles from IBZ to FBZ using symmetry operations.

    Handles proper rotation of dipole vectors and time-reversal conjugation.

    Parameters
    ----------
    dipoles_ibz : ndarray
        Dipoles at IBZ k-points with shape (nk_ibz, 3, nc, nv) or (nspin, nk_ibz, 3, nc, nv)
    latdb : YamboLatticeDB
        Lattice database with symmetry information
    time_rev_list : list, optional
        Time-reversal list for each symmetry. If None, computed from latdb

    Returns
    -------
    ndarray
        Dipoles expanded to FBZ with shape (nk_fbz, 3, nc, nv) or (nspin, nk_fbz, 3, nc, nv)
    """
    is_spinpol = dipoles_ibz.ndim == 5
    if is_spinpol:
        nspin, nk_ibz, _, nc, nv = dipoles_ibz.shape
        dipoles_fbz = np.zeros((latdb.nkBZ, 3, nc, nv), dtype=dipoles_ibz.dtype)
    else:
        nk_ibz, _, nc, nv = dipoles_ibz.shape
        dipoles_fbz = np.zeros((latdb.nkBZ, 3, nc, nv), dtype=dipoles_ibz.dtype)

    # Compute time-reversal list if not provided
    if time_rev_list is None:
        time_rev_list = latdb.time_rev_list if hasattr(latdb, 'time_rev_list') else \
                       [i >= len(latdb.sym_car) / (1 + int(latdb.time_rev))
                        for i in range(len(latdb.sym_car))]

    # Expand from IBZ to FBZ
    for i_fbz in range(latdb.nkBZ):
        i_ibz = latdb.kpoints_indexes[i_fbz]
        i_sym = latdb.symmetry_indexes[i_fbz]

        # Get rotation matrix
        rot_mat = latdb.sym_car[i_sym]

        # Get time-reversal flag
        trev = time_rev_list[i_sym]

        # Extract dipoles at this IBZ point
        if is_spinpol:
            d_ibz = dipoles_ibz[:, i_ibz, :, :, :]  # (nspin, 3, nc, nv)
            # Rotate for each spin
            for i_spin in range(nspin):
                dipoles_fbz[i_fbz, :, :, :] += rotate_dipole_vector(d_ibz[i_spin], rot_mat, trev) / nspin
        else:
            d_ibz = dipoles_ibz[i_ibz, :, :, :]  # (3, nc, nv)
            dipoles_fbz[i_fbz, :, :, :] = rotate_dipole_vector(d_ibz, rot_mat, trev)

    if is_spinpol:
        # Return with nspin dimension
        dipoles_fbz_final = np.zeros((nspin, latdb.nkBZ, 3, nc, nv), dtype=dipoles_ibz.dtype)
        for i_spin in range(nspin):
            dipoles_fbz_final[i_spin] = dipoles_fbz
        return dipoles_fbz_final

    return dipoles_fbz
