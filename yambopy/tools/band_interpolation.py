#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: RR, FP
#
# This file is part of the yambopy project
#
"""
Generic band/exciton interpolation utilities, agnostic to data source.
Works with any eigenvalues on k-grid: electronic bands, QP corrections, excitons, etc.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline, RBFInterpolator


def interp_cubic_spline_1d(scatter_x, scatter_energies, dense_x):
    """
    1D cubic spline interpolation along path distance coordinate, per band.

    Parameters
    ----------
    scatter_x : (N,) path distances of computed points
    scatter_energies : (N, nbands) energies at those points
    dense_x : (npts,) target path distances

    Returns
    -------
    dense_energies : (npts, nbands) interpolated energies
    """
    # Average duplicates (same x, different y from endpoint wrapping)
    unique_x, inv = np.unique(np.round(scatter_x, decimals=10), return_inverse=True)
    nbands = scatter_energies.shape[1]
    counts = np.bincount(inv, minlength=len(unique_x)).astype(float)
    unique_e = np.zeros((len(unique_x), nbands))
    for ib in range(nbands):
        unique_e[:, ib] = (
            np.bincount(inv, weights=scatter_energies[:, ib], minlength=len(unique_x)) / counts
        )

    if len(unique_x) < 3:
        raise ValueError(
            "Too few points on path for cubic spline (%d found, need >= 3). "
            "Increase tol or use method='nn'." % len(unique_x)
        )

    dense_e = np.zeros((len(dense_x), nbands))
    for ib in range(nbands):
        dense_e[:, ib] = CubicSpline(unique_x, unique_e[:, ib])(dense_x)
    return dense_e


def interp_rbf_2d(car_pts_source, eigens_source, car_pts_target, rlat, kernel='linear'):
    """
    RBF interpolation in 2D BZ plane (for 2D systems or k-plane slices).

    Parameters
    ----------
    car_pts_source : (nk, 3) Cartesian source k-points
    eigens_source : (nk, nbands) source eigenvalues
    car_pts_target : (npts, 3) Cartesian target k-points
    rlat : (3, 3) reciprocal lattice vectors
    kernel : 'linear' | 'thin_plate_spline' | 'multiquadric' | 'gaussian'

    Returns
    -------
    eigens_target : (npts, nbands) interpolated eigenvalues
    """
    def fold_to_2d(car_pts):
        """Fold to [-0.5, 0.5) for BZ-boundary consistency."""
        from yambopy.lattice import car_red, red_car
        red = car_red(car_pts, rlat)
        return red_car((red + 0.5) % 1.0 - 0.5, rlat)[:, :2]

    pts_2d = fold_to_2d(car_pts_source)
    target_2d = fold_to_2d(car_pts_target)
    nbands = eigens_source.shape[1]
    result = np.zeros((len(car_pts_target), nbands))

    try:
        for ib in range(nbands):
            result[:, ib] = RBFInterpolator(
                pts_2d, eigens_source[:, ib], kernel=kernel, smoothing=0.
            )(target_2d)
        return result
    except Exception as e:
        raise RuntimeError(f"RBF interpolation failed with kernel '{kernel}': {e}")


def interp_rbf_3d(car_pts_source, eigens_source, car_pts_target, kernels=None):
    """
    RBF interpolation in 3D k-space. Tries kernels in order until one succeeds.

    Parameters
    ----------
    car_pts_source : (nk, 3) Cartesian source k-points
    eigens_source : (nk, nbands) source eigenvalues
    car_pts_target : (npts, 3) Cartesian target k-points
    kernels : list of str, default ['linear', 'thin_plate_spline', 'multiquadric', 'gaussian']

    Returns
    -------
    eigens_target : (npts, nbands) interpolated eigenvalues
    """
    if kernels is None:
        kernels = ['linear', 'thin_plate_spline', 'multiquadric', 'gaussian']

    nbands = eigens_source.shape[1]
    result = np.zeros((len(car_pts_target), nbands))

    for kernel in kernels:
        try:
            for ib in range(nbands):
                result[:, ib] = RBFInterpolator(
                    car_pts_source, eigens_source[:, ib], kernel=kernel, smoothing=0.
                )(car_pts_target)
            print(f"RBF kernel '{kernel}' succeeded")
            return result
        except Exception as e:
            print(f"RBF kernel '{kernel}' failed: {e}")

    raise RuntimeError("All RBF kernels failed")


def interp_skw(red_kpts_ibz, eigens_ibz, red_kpts_target, lattice_cell, symrel,
               has_timrev=True, lpratio=6, nelect=1, fermie=0.0, verbose=0):
    """
    Star-function Wannier–Fourier (SKW) interpolation.

    Parameters
    ----------
    red_kpts_ibz : (nk_ibz, 3) IBZ k-points in reduced coords
    eigens_ibz : (nk_ibz, nbands) or (nk_ibz, nbands, nspin) eigenvalues
    red_kpts_target : (npts, 3) target k-points in reduced coords
    lattice_cell : (lat, red_atomic_positions, atomic_numbers) tuple
    symrel : (nsym, 3, 3) symmetry operations in reduced coords
    has_timrev : bool, time-reversal symmetry
    lpratio : int, SKW parameter
    nelect : int, number of electrons
    fermie : float, Fermi energy
    verbose : int, verbosity level

    Returns
    -------
    eigens_target : (npts, nbands) or (npts, nbands, nspin) interpolated eigenvalues
    """
    from yambopy.tools.skw import SkwInterpolator

    # Handle spin dimension
    if eigens_ibz.ndim == 2:
        eigens_ibz = eigens_ibz[np.newaxis, :, :]  # (1, nk, nbands) → (1, nk, nbands)
        has_spin = False
    else:
        # (nk, nbands, nspin) → (nspin, nk, nbands)
        eigens_ibz = np.moveaxis(eigens_ibz, -1, 0)
        has_spin = True

    skw = SkwInterpolator(
        lpratio=lpratio,
        kpts=red_kpts_ibz,
        eigens=eigens_ibz,
        fermie=fermie,
        nelect=nelect,
        cell=lattice_cell,
        symrel=symrel,
        has_timrev=has_timrev,
        verbose=verbose,
    )
    result = skw.interp_kpts(red_kpts_target).eigens

    if has_spin:
        return np.moveaxis(result, 0, -1)  # (nspin, npts, nbands) → (npts, nbands, nspin)
    else:
        return result[0]  # (1, npts, nbands) → (npts, nbands)


def interp_nn(red_kpts_ibz, eigens_ibz, red_kpts_target):
    """
    Nearest-neighbour interpolation on IBZ grid expanded to full BZ.

    Parameters
    ----------
    red_kpts_ibz : (nk_ibz, 3) IBZ k-points in reduced coords
    eigens_ibz : (nk_ibz, nbands) eigenvalues
    red_kpts_target : (npts, 3) target k-points in reduced coords

    Returns
    -------
    eigens_target : (npts, nbands) interpolated eigenvalues
    """
    _, idx = cKDTree(red_kpts_ibz).query(red_kpts_target)
    return eigens_ibz[idx]


def interp_band_along_1d_path(scatter_x, scatter_energies, dense_x, method='cubic_spline',
                              car_pts_source=None, car_pts_target=None, rlat=None,
                              red_kpts_ibz=None, red_kpts_target=None,
                              lattice_cell=None, symrel=None, **interp_kw):
    """
    Dispatch to appropriate 1D path interpolation method.

    Parameters
    ----------
    scatter_x : (N,) path distances of computed points
    scatter_energies : (N, nbands) energies at those points
    dense_x : (npts,) target path distances
    method : 'cubic_spline' | 'rbf_2d' | 'rbf_3d' | 'skw' | 'nn'
    car_pts_source, car_pts_target : for 'rbf_*' methods
    rlat : reciprocal lattice, for 'rbf_2d'
    red_kpts_ibz, red_kpts_target : for 'skw' and 'nn'
    lattice_cell, symrel : for 'skw'
    **interp_kw : method-specific kwargs (lpratio, nelect, kernel, etc.)

    Returns
    -------
    dense_energies : (npts, nbands) interpolated energies
    """
    if method == 'cubic_spline':
        return interp_cubic_spline_1d(scatter_x, scatter_energies, dense_x)
    elif method == 'rbf_2d':
        if car_pts_source is None or car_pts_target is None or rlat is None:
            raise ValueError("rbf_2d requires car_pts_source, car_pts_target, rlat")
        return interp_rbf_2d(car_pts_source, scatter_energies, car_pts_target, rlat,
                             kernel=interp_kw.get('kernel', 'linear'))
    elif method == 'rbf_3d':
        if car_pts_source is None or car_pts_target is None:
            raise ValueError("rbf_3d requires car_pts_source, car_pts_target")
        return interp_rbf_3d(car_pts_source, scatter_energies, car_pts_target,
                             kernels=interp_kw.get('kernels', None))
    elif method == 'skw':
        if red_kpts_ibz is None or red_kpts_target is None:
            raise ValueError("skw requires red_kpts_ibz, red_kpts_target")
        if lattice_cell is None or symrel is None:
            raise ValueError("skw requires lattice_cell, symrel")
        return interp_skw(red_kpts_ibz, scatter_energies, red_kpts_target,
                          lattice_cell, symrel,
                          has_timrev=interp_kw.get('has_timrev', True),
                          lpratio=interp_kw.get('lpratio', 6),
                          nelect=interp_kw.get('nelect', 1),
                          fermie=interp_kw.get('fermie', 0.0),
                          verbose=interp_kw.get('verbose', 0))
    elif method == 'nn':
        if red_kpts_ibz is None or red_kpts_target is None:
            raise ValueError("nn requires red_kpts_ibz, red_kpts_target")
        # For NN along path, we interpolate the scatter energies directly
        # (NN doesn't use the dense path in the same way as spline)
        return interp_nn(red_kpts_ibz, scatter_energies, red_kpts_target)
    else:
        raise ValueError(
            f"method must be 'cubic_spline', 'rbf_2d', 'rbf_3d', 'skw', or 'nn' — got '{method}'"
        )


def interp_spin_along_1d_path(scatter_x, scatter_spin, dense_x):
    """
    Interpolate per-band scalar (e.g., S_z) from scatter points onto dense path.
    Duplicated x-positions are averaged before spline fitting.

    Parameters
    ----------
    scatter_x : (N,) path distances
    scatter_spin : (N,) or (N, nbands) scalar values (e.g., S_z expectation)
    dense_x : (npts,) target path distances

    Returns
    -------
    dense_spin : (npts,) or (npts, nbands) interpolated scalars
    """
    if scatter_spin.ndim == 1:
        # Single band case
        unique_x, inv = np.unique(np.round(scatter_x, decimals=10), return_inverse=True)
        counts = np.bincount(inv, minlength=len(unique_x)).astype(float)
        unique_s = np.bincount(inv, weights=scatter_spin, minlength=len(unique_x)) / counts
        if len(unique_x) < 2:
            return np.full(len(dense_x), unique_s[0])
        return CubicSpline(unique_x, unique_s, extrapolate=True)(dense_x)
    else:
        # Multi-band case
        nbands = scatter_spin.shape[1]
        dense_spin = np.zeros((len(dense_x), nbands))
        for ib in range(nbands):
            dense_spin[:, ib] = interp_spin_along_1d_path(scatter_x, scatter_spin[:, ib], dense_x)
        return dense_spin
