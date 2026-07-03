# Copyright (c) 2026, University of Luxembourg
# All rights reserved.
#
# Author: RR
#
"""
Rotate degenerate exciton manifolds into the circular-polarization (sigma+/sigma-)
eigenbasis.

Iterative BSE diagonalizers (e.g. SLEPc) return an arbitrary orthonormal basis
within each (near-)degenerate eigenspace. For valley physics in hexagonal 2D
systems (K/K' doublets) this mixes the sigma+ and sigma- bright states, which
breaks any observable that is not invariant under rotations inside the
degenerate subspace (e.g. valley-resolved populations in exciton Bloch
equations, and Kerr rotation extracted from them).

This module builds a block-diagonal unitary, acting only inside degenerate
blocks, that maximizes the sigma+ / sigma- purity of the bright states. The
rotation must be applied ONCE, right after loading each ndb.BS_diago_QN and
before computing exciton dipoles or exciton-phonon matrix elements, so that
all downstream quantities live in the same, well-defined basis.

The construction follows the same pattern as exciton_spin.get_spinvals:
degenerate blocks are located with find_degeneracy_evs, and the physical
operator (here the sigma+ dipole projection) is resolved inside each block.
"""
import numpy as np
from yambopy.tools.degeneracy_finder import find_degeneracy_evs

# In-plane circular polarization vectors (light propagating along z)
E_PLUS  = np.array([1.0,  1.0j, 0.0]) / np.sqrt(2.0)
E_MINUS = np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0)


def circular_basis_matrix(eigenvalues, exc_dipoles, atol=1e-4, rtol=0.0,
                          dark_tol=1e-3, verbose=False):
    """
    Build the block-diagonal unitary U that rotates degenerate exciton blocks
    into the circular (sigma+/sigma-) eigenbasis.

    New states are |n'> = sum_m U[m, n'] |m>, i.e. columns of U are the
    coefficient vectors of the rotated states in the raw basis. Within each
    rotated bright block the ordering is [sigma+, sigma-, dark...].

    Parameters
    ----------
    eigenvalues : array (nexc,), real or complex
        Exciton energies (only the real part is used for degeneracy detection).
    exc_dipoles : array (3, nexc), complex
        Exciton dipole vectors D_n = sum_kcv A_n,kcv d_kcv.
    atol, rtol : float
        Degeneracy tolerances passed to find_degeneracy_evs (same units as
        eigenvalues).
    dark_tol : float
        Blocks whose total in-plane dipole strength is below
        dark_tol * max_n |D_n|^2 are considered dark and left untouched
        (a dark block's dipoles are numerical noise, so any basis in it is
        an equally valid gauge choice).
    verbose : bool
        Print a per-block purity report.

    Returns
    -------
    U : array (nexc, nexc), complex
        Block-diagonal unitary (identity outside degenerate bright blocks).
    blocks : list of dict
        One entry per rotated block: indices, sigma+/sigma- purities before
        and after rotation.
    """
    energies = np.asarray(eigenvalues).real
    D = np.asarray(exc_dipoles)
    nexc = len(energies)
    assert D.shape == (3, nexc), f"exc_dipoles must be (3, nexc), got {D.shape}"

    U = np.eye(nexc, dtype=complex)
    blocks = []

    Dmax2 = (np.abs(D)**2).sum(axis=0).max()
    degen_sets = find_degeneracy_evs(energies, atol=atol, rtol=rtol)

    for ids in degen_sets:
        ids = np.sort(np.asarray(ids))
        n = len(ids)
        if n < 2: continue

        Dblk = D[:, ids]                       # (3, n)
        # Hermitian sigma+ / sigma- projection amplitudes of each raw state:
        # p_m = <e+|D_m>, q_m = <e-|D_m>
        p = Dblk.T @ np.conj(E_PLUS)           # (n,)
        q = Dblk.T @ np.conj(E_MINUS)
        strength = (np.abs(p)**2 + np.abs(q)**2).sum()
        if strength < dark_tol * max(Dmax2, 1e-300):
            continue                           # dark block: leave as is

        # Maximal-sigma+ combination: c_R = conj(p)/|p|.
        # The conjugation is required (Cauchy-Schwarz: |sum_i c_i p_i| is
        # maximal for c = conj(p)/|p|); verified numerically in
        # yambopy_circular_basis_TODO.md (100% vs 99.64% purity).
        Ublk = np.zeros((n, n), dtype=complex)
        ncol = 0
        if np.linalg.norm(p) > 0:
            Ublk[:, ncol] = np.conj(p) / np.linalg.norm(p)
            ncol += 1
        # Maximal-sigma- combination, orthogonalized against the sigma+ one
        if np.linalg.norm(q) > 0:
            cL = np.conj(q) / np.linalg.norm(q)
            for j in range(ncol):
                cL -= Ublk[:, j] * np.vdot(Ublk[:, j], cL)
            nrm = np.linalg.norm(cL)
            if nrm > 1e-8:
                Ublk[:, ncol] = cL / nrm
                ncol += 1
        # Fill the remaining (dark) columns by Gram-Schmidt on the identity
        for i in range(n):
            if ncol == n: break
            c = np.zeros(n, dtype=complex); c[i] = 1.0
            for j in range(ncol):
                c -= Ublk[:, j] * np.vdot(Ublk[:, j], c)
            nrm = np.linalg.norm(c)
            if nrm > 1e-8:
                Ublk[:, ncol] = c / nrm
                ncol += 1
        assert ncol == n, "failed to build a complete unitary block"

        U[np.ix_(ids, ids)] = Ublk

        # purity report: fraction of in-plane strength in the dominant channel
        Drot = Dblk @ Ublk
        def purity(dd):
            pp, qq = abs(np.vdot(E_PLUS, dd))**2, abs(np.vdot(E_MINUS, dd))**2
            tot = pp + qq
            return max(pp, qq)/tot if tot > 0 else 0.0
        info = dict(indices=ids.tolist(),
                    purity_before=[purity(Dblk[:, i]) for i in range(n)],
                    purity_after=[purity(Drot[:, i]) for i in range(n)])
        blocks.append(info)
        if verbose:
            print(f"[circular_basis] block {ids.tolist()} "
                  f"E={energies[ids[0]]:.6f}: purity "
                  f"{['%.4f' % x for x in info['purity_before']]} -> "
                  f"{['%.4f' % x for x in info['purity_after']]}")

    return U, blocks


def rotate_excdb_to_circular(excdb, dipdb=None, bands_range=None, atol=1e-4,
                             rtol=0.0, dark_tol=1e-3, verbose=False):
    """
    Rotate a YamboExcitonDB, in place, to the circular basis inside each
    degenerate manifold. Meant to be called right after from_db_file on the
    Q=Gamma database, before any dipole / exciton-phonon calculation.

    Rotates consistently: Akcv, the flat eigenvectors, exc_dipoles and the
    optical residuals. Since all downstream quantities (exciton dipoles,
    exc-ph matrix elements) are linear in Akcv, rotating Akcv here makes the
    whole pipeline consistent with no further changes.

    Parameters
    ----------
    excdb : YamboExcitonDB
    dipdb : YamboDipolesDB, optional
        Needed if excdb.exc_dipoles has not been computed yet
        (compute_exciton_dipoles is called with expand=False dipoles).
    atol, rtol : float
        Degeneracy tolerances in the units of excdb.eigenvalues (eV).

    Returns
    -------
    U : (nexc, nexc) complex unitary actually applied.
    blocks : per-block purity report from circular_basis_matrix.
    """
    if excdb.exc_dipoles is None:
        if dipdb is None:
            raise ValueError("excdb has no exc_dipoles: pass dipdb to compute them")
        excdb.compute_exciton_dipoles(dipdb, bands_range=bands_range)

    U, blocks = circular_basis_matrix(excdb.eigenvalues, excdb.exc_dipoles,
                                      atol=atol, rtol=rtol, dark_tol=dark_tol,
                                      verbose=verbose)
    if not blocks:
        return U, blocks

    # |n'> = sum_m U[m,n'] |m>  =>  A'_n' = sum_m U[m,n'] A_m (linear in A)
    Akcv = excdb.get_Akcv()
    Akcv_rot = np.tensordot(U.T, Akcv, axes=(1, 0))
    excdb.Akcv = np.ascontiguousarray(Akcv_rot)
    if excdb.eigenvectors is not None:
        excdb.eigenvectors = excdb.flatten_Akcv(excdb.Akcv)

    # Dipoles are linear in A: D'_n' = sum_m U[m,n'] D_m
    excdb.exc_dipoles = excdb.exc_dipoles @ U

    # Optical residuals: r_n is linear in A_n, l_n = conj(r_n) in TDA,
    # so r' = U^T r and l' = conj(U)^T l keep |l' r'| consistent.
    if getattr(excdb, 'r_residual', None) is not None:
        excdb.r_residual = U.T @ excdb.r_residual
    if getattr(excdb, 'l_residual', None) is not None:
        excdb.l_residual = np.conj(U).T @ excdb.l_residual

    return U, blocks
