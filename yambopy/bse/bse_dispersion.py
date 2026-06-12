#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: FP, RR
#
# This file is part of the yambopy project
#
import os
from glob import glob
from qepy.lattice import Path
from yambopy import *
from yambopy.units import *
from yambopy.plot.plotting import add_fig_kwargs, BZ_Wigner_Seitz, shifted_grids_2D
from yambopy.lattice import replicate_red_kmesh, calculate_distances, car_red, red_car
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.kpoints import get_path
from yambopy.tools.band_interpolation import (
    interp_cubic_spline_1d, interp_rbf_2d, interp_skw, interp_nn,
    interp_spin_along_1d_path
)
import matplotlib.pyplot as plt
import numpy as np


class ExcitonDispersion():
    """
    Exciton band structure at finite momentum Q.

    Reads ndb.BS_diago_Q* databases and provides:
      - Scatter plot of computed Q-points along a path (no interpolation)
      - Interpolated band structure (cubic spline, RBF, SKW, or NN)
      - Spin expectation values along the path

    Parameters
    ----------
    lattice           : YamboLatticeDB
    nexcitons         : int, optional — exciton eigenstates to load (default: all)
    folder            : str — folder containing ndb.BS_diago_Q* files
    load_eigenvectors : bool — load A_kcv coefficients (required for spin)

    Attributes
    ----------
    ntransitions       : BSE basis size  (nk * nv * nc)
    nexcitons_available: eigenstates stored in the databases
    nexcitons          : eigenstates actually loaded
    """

    def __init__(self, lattice, nexcitons=None, folder='.', load_eigenvectors=False):
        if not isinstance(lattice, YamboLatticeDB):
            raise ValueError('lattice must be a YamboLatticeDB instance')

        files    = glob(folder + '/ndb.BS_diago_Q*')
        nqpoints = len(files)
        if nqpoints != lattice.ibz_nkpoints:
            raise ValueError("Incomplete Q-point list (%d / %d)" % (nqpoints, lattice.ibz_nkpoints))

        self._db_check(lattice, nqpoints, folder)

        if nexcitons is None:
            nexcitons = self.nexcitons_available
        elif nexcitons > self.nexcitons_available:
            raise ValueError(
                "Requested %d excitons but only %d available" % (nexcitons, self.nexcitons_available)
            )

        car_qpoints      = np.zeros((nqpoints, 3))
        exc_energies     = np.zeros((nqpoints, nexcitons))
        exc_eigenvectors = (
            np.zeros((nqpoints, nexcitons, self.ntransitions), dtype=complex)
            if load_eigenvectors else None
        )

        for iQ in range(nqpoints):
            exc_obj = YamboExcitonDB.from_db_file(
                lattice, filename=folder + '/ndb.BS_diago_Q%d' % (iQ + 1),
                Load_WF=load_eigenvectors
            )
            car_qpoints[iQ]  = np.array([0., 0., 0.]) if iQ == 0 else exc_obj.car_qpoint
            exc_energies[iQ] = exc_obj.eigenvalues[:nexcitons].real
            if load_eigenvectors:
                exc_eigenvectors[iQ] = exc_obj.eigenvectors[:nexcitons]

        self.folder           = folder
        self.nqpoints         = nqpoints
        self.nexcitons        = nexcitons
        self.car_qpoints      = car_qpoints
        self.red_qpoints      = car_red(car_qpoints, lattice.rlat)
        self.lattice          = lattice
        self.exc_energies     = exc_energies
        self.exc_eigenvectors = exc_eigenvectors
        self.alat             = lattice.alat
        self.rlat             = lattice.rlat

    # ------------------------------------------------------------------
    # Database validation
    # ------------------------------------------------------------------

    def _db_check(self, lattice, nqpoints, folder):
        """Validate ntransitions and nexcitons consistency across Q-points."""
        nexcitons_list    = np.zeros(nqpoints, dtype=int)
        ntransitions_list = np.zeros(nqpoints, dtype=int)

        for iQ in range(nqpoints):
            exc_obj = YamboExcitonDB.from_db_file(
                lattice, filename=folder + '/ndb.BS_diago_Q%d' % (iQ + 1),
                Load_WF=False
            )
            nexcitons_list[iQ]    = exc_obj.nexcitons     # number of solved eigenstates
            ntransitions_list[iQ] = exc_obj.ntransitions  # BSE basis size = nk*nv*nc
            if iQ == 0:
                tbl = exc_obj.table

        if not np.all(ntransitions_list == ntransitions_list[0]):
            raise ValueError("BSE basis size (ntransitions) is inconsistent across Q-points.")

        if not np.all(nexcitons_list == nexcitons_list[0]):
            print("[WARNING] Number of exciton eigenstates differs across Q-points. Taking minimum.")

        self.nexcitons_available = int(np.min(nexcitons_list))
        self.ntransitions        = int(ntransitions_list[0])
        self.nkpoints            = int(np.max(tbl[:, 0]))
        self.nvalence            = len(np.unique(tbl[:, 1]))
        self.nconduction         = len(np.unique(tbl[:, 2]))
        self.exc_table           = tbl

        is_spin_pol = len(np.unique(tbl[:, 3])) > 1 or len(np.unique(tbl[:, 4])) > 1
        if is_spin_pol:
            print("[WARNING] Spin-polarised system detected (partially supported)")

    # ------------------------------------------------------------------
    # Eigenvector reshape (optional, needed for spin)
    # ------------------------------------------------------------------

    def reshape_eigenvectors(self):
        """
        Return eigenvectors reshaped from [nq, nexcitons, ntransitions]
        to [nq, nexcitons, nkpoints, nvalence, nconduction].
        Requires load_eigenvectors=True.
        """
        if self.exc_eigenvectors is None:
            raise RuntimeError("Eigenvectors not loaded. Use load_eigenvectors=True.")
        nq, nexc = self.nqpoints, self.nexcitons
        nk, nv, nc = self.nkpoints, self.nvalence, self.nconduction
        return self.exc_eigenvectors.reshape(nq, nexc, nk, nv, nc)

    # ------------------------------------------------------------------
    # BZ weight plot
    # ------------------------------------------------------------------

    @add_fig_kwargs
    def plot_Aweights(self, data, plt_show=False, plt_cbar=False, **kwargs):
        """
        2D scatter in the IBZ q-BZ of any scalar quantity on the q-grid.
        NB: requires a 2D system.
        """
        if len(data) != len(self.car_qpoints):
            raise ValueError(
                'data length (%d) != number of qpoints (%d)' % (len(data), len(self.car_qpoints))
            )

        fig, ax = plt.subplots(1, 1)
        c_BZ = 'black' if self.nqpoints < self.nkpoints else 'white'
        ax.add_patch(BZ_Wigner_Seitz(self.lattice, color=c_BZ))
        lim = 1.05 * np.linalg.norm(self.rlat[0])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        BZs = shifted_grids_2D(self.car_qpoints, self.rlat)
        for qpts_s in BZs:
            plot = ax.scatter(qpts_s[:, 0], qpts_s[:, 1], c=data, **kwargs)

        if plt_cbar:
            fig.colorbar(plot)
        plt.gca().set_aspect('equal')
        if plt_show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Path geometry helpers
    # ------------------------------------------------------------------

    def _path_geometry(self, path):
        """
        Extract segment endpoints (Cartesian) and cumulative boundary distances
        from a Path object.

        Returns
        -------
        seg_ends   : (n_seg+1, 3) Cartesian endpoints
        seg_lens   : (n_seg,) length of each segment in Ang^-1
        boundaries : (n_seg+1,) cumulative distances at high-symmetry points
        labels     : list of high-symmetry labels
        """
        klist    = path.get_klist()
        car_kpts = red_car(klist[:, :3], self.rlat)

        boundary_idx = [0]
        idx = 0
        for npts in path.intervals:
            idx += npts
            boundary_idx.append(idx)

        seg_ends   = car_kpts[np.array(boundary_idx)]
        seg_lens   = np.linalg.norm(np.diff(seg_ends, axis=0), axis=1)
        boundaries = np.concatenate([[0.], np.cumsum(seg_lens)])
        return seg_ends, seg_lens, boundaries, path.klabels

    def _dense_path(self, path, npts=300):
        """
        Generate `npts` points uniformly spaced along the path.

        Returns
        -------
        dense_x   : (npts,) 1D path distances
        dense_car : (npts, 3) Cartesian coordinates
        dense_red : (npts, 3) reduced coordinates
        """
        seg_ends, _, boundaries, _ = self._path_geometry(path)
        dense_x   = np.linspace(0., boundaries[-1], npts)
        dense_car = np.array([
            np.interp(dense_x, boundaries, seg_ends[:, i]) for i in range(3)
        ]).T
        return dense_x, dense_car, car_red(dense_car, self.rlat)

    def _project_qpts_onto_path(self, path, tol=1e-3, expand_bz=True):
        """
        Project Q-points onto the path, collecting every point within `tol`
        (Cartesian Ang^-1) of any segment.

        Two issues handled explicitly:

        * BZ-wrapping: the full-BZ Q-points from expand_kpoints are stored in raw
          Cartesian (sym * k), which can differ from the path convention by a
          reciprocal lattice vector.  All 27 first-shell periodic images are tried
          so that, e.g., M at (0, -0.5, 0) is matched to the path point M at
          (0, 0.5, 0) via a G-vector shift.

        * Endpoint duplicates: a point like Gamma that sits at BOTH the start and
          end of a path \Gamma→…→\Gamma is added at each boundary independently, so it
          appears in the scatter/spline at x=0 AND x=total_length.

        When expand_bz=False only the directly-computed IBZ Q-points are considered.
        When expand_bz=True (default) the full BZ is used and is_ibz marks which
        points were directly computed vs symmetry-expanded.

        Returns
        -------
        path_coords : (N,) 1D distances along path
        q_indices   : (N,) indices into the full-BZ Q-point array
        is_ibz      : (N,) bool — True = directly computed, False = symmetry-expanded
        boundaries  : (n_seg+1,) cumulative distances at high-symmetry points
        labels      : high-symmetry labels
        """
        from scipy.spatial import cKDTree

        seg_ends, seg_lens, boundaries, labels = self._path_geometry(path)
        car_qpoints_full = red_car(self.lattice.red_kpoints, self.rlat)

        # IBZ detection via coordinate comparison (robust for all symmetry orderings
        # and for special points like Gamma that are invariant under every operation).
        ibz_tree     = cKDTree(self.car_qpoints)
        dists_ibz, _ = ibz_tree.query(car_qpoints_full)
        is_ibz_full  = dists_ibz < 1e-4  # fixed small tolerance for identity check

        # First-shell reciprocal-lattice images to handle BZ-wrapping mismatches
        images = np.array([
            i * self.rlat[0] + j * self.rlat[1] + k * self.rlat[2]
            for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
        ])

        path_coords, q_indices, is_ibz = [], [], []

        for iq, qpt in enumerate(car_qpoints_full):
            if not expand_bz and not is_ibz_full[iq]:
                continue

            # Collect all distinct path coordinates this Q-point maps to
            hits = {}  # rounded_coord -> exact coord

            for G in images:
                qpt_img = qpt + G

                # Segment endpoints — checked exactly so that Gamma (or any HS
                # point) registers at EVERY boundary where it sits.
                for i, hs_pt in enumerate(seg_ends):
                    if np.linalg.norm(qpt_img - hs_pt) < tol:
                        key = round(boundaries[i], 12)
                        hits[key] = boundaries[i]

                # Segment interiors — t strictly in (0, 1) to avoid double-counting
                # with the endpoint check above.
                for i, (A, B) in enumerate(zip(seg_ends[:-1], seg_ends[1:])):
                    if seg_lens[i] < 1e-10:
                        continue
                    seg = B - A
                    t   = np.dot(qpt_img - A, seg) / seg_lens[i]**2
                    if t <= 1e-10 or t >= 1. - 1e-10:
                        continue
                    perp = np.linalg.norm(qpt_img - A - t * seg)
                    if perp < tol:
                        coord = boundaries[i] + t * seg_lens[i]
                        key   = round(coord, 10)
                        hits[key] = coord

            for coord in hits.values():
                path_coords.append(coord)
                q_indices.append(iq)
                is_ibz.append(is_ibz_full[iq])

        if not path_coords:
            msg = "No Q-points found within tol=%.4g" % tol
            if not expand_bz:
                msg += ". Try expand_bz=True to include symmetry-equivalent points"
            raise ValueError(msg + ". Check path alignment or increase tolerance.")

        path_coords = np.array(path_coords)
        q_indices   = np.array(q_indices, dtype=int)
        is_ibz      = np.array(is_ibz, dtype=bool)
        order       = np.argsort(path_coords)
        return path_coords[order], q_indices[order], is_ibz[order], boundaries, labels

    # ------------------------------------------------------------------
    # Interpolation (via generalized utilities in tools.band_interpolation)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dispersion (scatter — no interpolation)
    # ------------------------------------------------------------------

    def get_dispersion(self, path, tol=1e-3, expand_bz=True):
        """
        Return Q-points along the path and their energies.

        Parameters
        ----------
        path      : Path object
        tol       : maximum Cartesian distance (Ang^-1) to accept a Q-point
        expand_bz : if True, include symmetry-equivalent Q-points beyond the IBZ

        Returns
        -------
        path_coords : (N,) 1D distances along path
        energies    : (N, nexcitons)
        is_ibz      : (N,) bool — True = directly computed, False = symmetry-expanded
        boundaries  : (n_seg+1,) cumulative distances at high-symmetry points
        labels      : list of high-symmetry labels
        """
        path_coords, q_indices, is_ibz, boundaries, labels = \
            self._project_qpts_onto_path(path, tol=tol, expand_bz=expand_bz)
        energies_full = self.exc_energies[self.lattice.kpoints_indexes]
        return path_coords, energies_full[q_indices], is_ibz, boundaries, labels

    # ------------------------------------------------------------------
    # Dispersion (interpolated)
    # ------------------------------------------------------------------

    def get_dispersion_interpolated(self, path, method='cubic_spline', npts=300, tol=1e-3,
                                    lpratio=6, nelect=1, expand_bz=True):
        """
        Interpolate the exciton dispersion along the path.

        Parameters
        ----------
        path      : Path object
        method    : 'cubic_spline' | 'rbf' | 'skw' | 'nn'
        npts      : number of dense points for the interpolated line
        tol       : tolerance for projecting Q-points onto path (scatter overlay)
        expand_bz : include symmetry-equivalent Q-points in the scatter overlay
        lpratio, nelect : SKW-specific parameters

        Returns
        -------
        dense_x          : (npts,) path coordinates
        dense_energies   : (npts, nexcitons) interpolated energies
        scatter_x        : (N,) path coordinates of Q-points used for scatter overlay
        scatter_energies : (N, nexcitons) energies at those Q-points
        is_ibz           : (N,) bool — True = directly computed, False = symmetry-expanded
        boundaries       : (n_seg+1,) cumulative distances at high-symmetry points
        labels           : list of high-symmetry labels
        """
        scatter_x, q_indices, is_ibz, boundaries, labels = \
            self._project_qpts_onto_path(path, tol=tol, expand_bz=expand_bz)
        energies_full    = self.exc_energies[self.lattice.kpoints_indexes]
        scatter_energies = energies_full[q_indices]

        dense_x, dense_car, dense_red = self._dense_path(path, npts)

        if method == 'cubic_spline':
            dense_energies = interp_cubic_spline_1d(scatter_x, scatter_energies, dense_x)
        elif method == 'rbf':
            car_qpts_full = red_car(self.lattice.red_kpoints, self.rlat)
            dense_energies = interp_rbf_2d(car_qpts_full, energies_full, dense_car, self.rlat)
        elif method == 'skw':
            latnp = self.lattice.lat
            sym_rel = np.stack([
                np.round(np.linalg.inv(latnp) @ s @ latnp)
                for s in self.lattice.sym_car
            ])
            dense_energies = interp_skw(
                self.red_qpoints, self.exc_energies, dense_red,
                (self.lattice.lat, self.lattice.red_atomic_positions, self.lattice.atomic_numbers),
                sym_rel, has_timrev=bool(self.lattice.time_rev),
                lpratio=lpratio, nelect=nelect, verbose=0
            )
        elif method == 'nn':
            dense_energies = interp_nn(self.lattice.red_kpoints, energies_full, dense_red)
        else:
            raise ValueError(
                "method must be 'cubic_spline', 'rbf', 'skw', or 'nn' — got '%s'" % method
            )

        return dense_x, dense_energies, scatter_x, scatter_energies, is_ibz, boundaries, labels

    # ------------------------------------------------------------------
    # Spin
    # ------------------------------------------------------------------

    def _make_wfdb(self, save_dir, bands_range):
        """
        Instantiate YamboWFDB from a SAVE path that may be bare ('SAVE') or nested
        ('run/SAVE', '/abs/path/SAVE').  YamboWFDB expects path=<parent> and
        save=<folder_name> separately, so we split here.
        """
        from yambopy.dbs.wfdb import YamboWFDB
        parent    = os.path.dirname(save_dir) or '.'
        save_name = os.path.basename(save_dir) or save_dir
        return YamboWFDB(path=parent, save=save_name, latdb=self.lattice, bands_range=bands_range)

    def _compute_spin_full_bz(self, save_dir, bse_dir, contribution,
                               sz=0.5 * np.array([[1, 0], [0, -1]]),
                               dmat_mode='run', dmat_file='Dmats.npy'):
        """
        Compute S_z at all full-BZ Q-points by explicitly rotating the exciton
        wavefunctions to each symmetry-equivalent Q-point.
        Returns spin_full of shape (nq_full, nexcitons).
        """
        from yambopy.bse.exciton_spin import compute_exciton_spin, get_spinvals
        from yambopy.exciton_phonon.excph_matrix_elements import rotate_Akcv_Q, save_or_load_dmat

        excdb_q1 = YamboExcitonDB.from_db_file(
            self.lattice, filename='ndb.BS_diago_Q1',
            folder=bse_dir, Load_WF=True, neigs=self.nexcitons
        )
        bands_range = [np.min(excdb_q1.table[:, 1]) - 1, np.max(excdb_q1.table[:, 2])]
        wfdb    = self._make_wfdb(save_dir, bands_range)
        elec_sz = wfdb.get_spin_m_e_BZ(s_z=sz)

        exdbs = [excdb_q1] + [
            YamboExcitonDB.from_db_file(
                self.lattice, filename='ndb.BS_diago_Q%d' % (iq + 1),
                folder=bse_dir, Load_WF=True, neigs=self.nexcitons
            )
            for iq in range(1, self.nqpoints)
        ]

        nq_full   = len(self.lattice.kpoints_indexes)
        spin_full = np.zeros((nq_full, self.nexcitons))
        Dmats     = save_or_load_dmat(wfdb, mode=dmat_mode, dmat_file=dmat_file)

        for iq_full in range(nq_full):
            Qpt    = self.lattice.red_kpoints[iq_full]
            rot_Ak = rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats, folder=None)
            iq_ibz = self.lattice.kpoints_indexes[iq_full]
            excdb  = exdbs[iq_ibz]

            original_get_Akcv = excdb.get_Akcv
            excdb.get_Akcv    = lambda: rot_Ak
            smat = compute_exciton_spin(
                self.lattice, excdb, wfdb, elec_sz, contribution=contribution, diagonal=False
            )
            smat = get_spinvals(smat, excdb.eigenvalues, atol=1e-2)
            excdb.get_Akcv = original_get_Akcv

            vals = [v for group in smat for v in group]
            spin_full[iq_full] = np.array(vals)[:self.nexcitons].real

        return spin_full

    def get_spin_along_path(self, path, tol=1e-3, expand_bz=True,
                            save_dir='SAVE', bse_dir='BSE', contribution='b',
                            dmat_mode='run', dmat_file='Dmats.npy'):
        """
        Compute S_z expectation values at the Q-points that lie along the path.

        Parameters
        ----------
        expand_bz : must match the value used in get_dispersion / get_dispersion_interpolated
                    so that the returned array aligns with the scatter data

        Returns
        -------
        spin : (N, nexcitons) where N matches the scatter points from get_dispersion*
        """
        if self.lattice.spinor_components == 1:
            raise ValueError(
                "Spin-projected exciton dispersion requires a spinor (non-collinear) "
                "calculation (spinor_components=2). For collinear systems the exciton "
                "spin is not a meaningful observable."
            )
        _, q_indices, _, _, _ = self._project_qpts_onto_path(path, tol=tol, expand_bz=expand_bz)
        spin_full = self._compute_spin_full_bz(
            save_dir, bse_dir, contribution,
            dmat_mode=dmat_mode, dmat_file=dmat_file
        )
        return spin_full[q_indices]

    def _interp_spin_dense(self, scatter_x, spin_band, dense_x):
        """Interpolate per-band S_z from scatter Q-points onto the dense path grid."""
        return interp_spin_along_1d_path(scatter_x, spin_band, dense_x)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_exciton_dispersion(self, path, interpolate=False, method='cubic_spline',
                                npts=300, tol=1e-3, expand_bz=True,
                                ylim=None, figsize=(8, 5), title="Exciton dispersion",
                                spin_data=None, lpratio=6, nelect=1, s=80,
                                alpha=0.7, return_data=False, **scatter_kw):
        """
        Plot the exciton dispersion.

        Parameters
        ----------
        path        : Path object
        interpolate : False → big filled dots at Q-points (no line)
                      True  → interpolated line + empty dots at Q-points
        method      : 'cubic_spline' (default) | 'rbf' | 'skw' | 'nn'
        npts        : number of dense points for the interpolated line
        tol         : tolerance for projecting Q-points onto path (Ang^-1)
        expand_bz   : include symmetry-expanded Q-points (shown as triangles ▲)
        ylim        : (ymin, ymax) energy window in eV
        spin_data   : (N, nexcitons) S_z values from get_spin_along_path()
                      — dots colored red (up) / blue (down), norm vmin=-0.5 vmax=0.5
                      — interpolated line colored by S_z splined onto the dense grid
        s           : marker size
        alpha       : marker transparency (default 0.7); pass 1.0 for fully opaque
        return_data : if True, return data dict instead of (fig, ax)
        scatter_kw  : extra keyword arguments forwarded to ax.scatter
        lpratio, nelect : SKW parameters

        Returns
        -------
        if return_data=False: (fig, ax)
        if return_data=True : (fig, ax, data) where data is dict with keys:
            Main path data (dense interpolated if interpolate=True, else scatter):
            'k_points'         : (npts, 3) k-points in reciprocal coordinates
            'distances'        : (npts,) path distances
            'energies'         : (npts, nexcitons) band energies

            Scatter data (computed Q-points):
            'scatter_k_points' : (N, 3) k-points of computed Q-points (reciprocal)
            'scatter_distances': (N,) path distances of computed Q-points
            'scatter_energies' : (N, nexcitons) energies at computed Q-points
            'is_ibz'           : (N,) bool array marking IBZ vs expanded points

            Path geometry:
            'boundaries'       : (n_seg+1,) path distances at high-symmetry points
            'labels'           : list of high-symmetry point labels
        """
        fig, ax = plt.subplots(figsize=figsize)

        spin_norm = plt.Normalize(vmin=-0.5, vmax=0.5)
        spin_cmap = plt.cm.bwr  # blue = spin down (−0.5), red = spin up (+0.5)

        def _scatter_band(x, y, is_ibz, color=None, c=None):
            """Draw computed points as circles, expanded points as triangles."""
            kw_shared = dict(s=s, linewidths=0, zorder=3, alpha=alpha, **scatter_kw)
            for mask, marker in [(is_ibz, 'o'), (~is_ibz, '^')]:
                if not mask.any():
                    continue
                if c is not None:
                    ax.scatter(x[mask], y[mask], c=c[mask],
                               cmap=spin_cmap, norm=spin_norm,
                               marker=marker, **kw_shared)
                else:
                    ax.scatter(x[mask], y[mask], color=color,
                               marker=marker, **kw_shared)

        if interpolate:
            dense_x, dense_e, scatter_x, scatter_e, is_ibz, boundaries, labels = \
                self.get_dispersion_interpolated(
                    path, method=method, npts=npts, tol=tol, expand_bz=expand_bz,
                    lpratio=lpratio, nelect=nelect
                )
            if spin_data is not None and spin_data.shape[0] != len(scatter_x):
                raise ValueError(
                    "spin_data has %d points but scatter has %d. "
                    "Call get_spin_along_path with the same tol and expand_bz=%s."
                    % (spin_data.shape[0], len(scatter_x), expand_bz)
                )
            if spin_data is not None:
                from matplotlib.collections import LineCollection
                for ib in range(self.nexcitons):
                    ax.plot(dense_x, dense_e[:, ib], color='black', lw=2.5,
                            ls='dotted', zorder=1)
                for ib in range(self.nexcitons):
                    # Interpolate spin onto the dense grid so the line tracks S_z
                    dense_spin = self._interp_spin_dense(scatter_x, spin_data[:, ib], dense_x)
                    pts  = np.array([dense_x, dense_e[:, ib]]).T.reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc   = LineCollection(segs, cmap=spin_cmap, norm=spin_norm, lw=2.5, zorder=2)
                    lc.set_array(dense_spin)
                    ax.add_collection(lc)
                for ib in range(self.nexcitons):
                    _scatter_band(scatter_x, scatter_e[:, ib], is_ibz,
                                  c=spin_data[:, ib])
                sm = plt.cm.ScalarMappable(cmap=spin_cmap, norm=spin_norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label(r"$\langle S_z \rangle$")
                cbar.set_ticks([-0.5, 0., 0.5])
            else:
                for ib in range(self.nexcitons):
                    ax.plot(dense_x, dense_e[:, ib], color='black', lw=2.5, zorder=1)
                for ib in range(self.nexcitons):
                    # empty dots: use facecolors='none' manually instead of _scatter_band
                    for mask, marker in [(is_ibz, 'o'), (~is_ibz, '^')]:
                        if not mask.any():
                            continue
                        ax.scatter(scatter_x[mask], scatter_e[mask, ib],
                                   s=s, marker=marker, zorder=3, alpha=alpha,
                                   facecolors='none', edgecolors='black', linewidths=1.5,
                                   **scatter_kw)
        else:
            scatter_x, scatter_e, is_ibz, boundaries, labels = \
                self.get_dispersion(path, tol=tol, expand_bz=expand_bz)

            if spin_data is not None and spin_data.shape[0] != len(scatter_x):
                raise ValueError(
                    "spin_data has %d points but scatter has %d. "
                    "Call get_spin_along_path with the same tol and expand_bz=%s."
                    % (spin_data.shape[0], len(scatter_x), expand_bz)
                )
            if spin_data is not None:
                for ib in range(self.nexcitons):
                    _scatter_band(scatter_x, scatter_e[:, ib], is_ibz,
                                  c=spin_data[:, ib])
                sm = plt.cm.ScalarMappable(cmap=spin_cmap, norm=spin_norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label(r"$\langle S_z \rangle$")
                cbar.set_ticks([-0.5, 0., 0.5])
            else:
                for ib in range(self.nexcitons):
                    _scatter_band(scatter_x, scatter_e[:, ib], is_ibz, color='black')

        for x in boundaries:
            ax.axvline(x, color='gray', lw=1.5, ls='--')
        ax.set_xticks(boundaries)
        ax.set_xticklabels(labels)
        ax.set_xlim(boundaries[0], boundaries[-1])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("Exciton energy (eV)")
        ax.set_title(title)
        plt.tight_layout()

        if return_data:
            car_qpts_full = red_car(self.lattice.red_kpoints, self.rlat)
            scatter_x, scatter_q_indices, is_ibz, boundaries, labels = \
                self._project_qpts_onto_path(path, tol=tol, expand_bz=expand_bz)
            scatter_car = car_qpts_full[scatter_q_indices]
            scatter_red = car_red(scatter_car, self.rlat)
            energies_full = self.exc_energies[self.lattice.kpoints_indexes]
            scatter_e = energies_full[scatter_q_indices]

            if interpolate:
                dense_x, dense_e, _, _, _, _, _ = \
                    self.get_dispersion_interpolated(
                        path, method=method, npts=npts, tol=tol, expand_bz=expand_bz,
                        lpratio=lpratio, nelect=nelect
                    )
                _, dense_car, dense_red = self._dense_path(path, npts)
            else:
                dense_x, dense_car, dense_red = self._dense_path(path, npts)
                dense_e = None

            data = {
                'scatter_k_points': scatter_red,
                'scatter_distances': scatter_x,
                'scatter_energies': scatter_e,
                'is_ibz': is_ibz,
                'boundaries': boundaries,
                'labels': labels,
            }

            if interpolate:
                data['k_points'] = dense_red
                data['distances'] = dense_x
                data['energies'] = dense_e
            else:
                data['k_points'] = scatter_red
                data['distances'] = scatter_x
                data['energies'] = scatter_e

            return fig, ax, data

        return fig, ax

    # ------------------------------------------------------------------
    # Orbital-projected exciton dispersion
    # ------------------------------------------------------------------

    def get_orbital_weights(self, projwfc, selected_orbitals, contribution='both'):
        """
        Compute orbital-projected exciton weights.

        For each Q-point and exciton state S, computes:
            W^S(Q) = sum_{kvc} |A^S_{kvc}(Q)|^2 * w_orb(k, v, c)
        where w_orb is the sum of |<psi_kn|phi_o>|^2 over selected_orbitals,
        with n = v, c, or their average depending on `contribution`.

        Parameters
        ----------
        projwfc           : ProjwfcXML — QE orbital projections (same k-grid as yambo)
        selected_orbitals : list of int — state indices (0-based, from get_states_helper)
        contribution      : 'valence' | 'conduction' | 'both' (default)

        Returns
        -------
        weights : (nqpoints, nexcitons) float array
        """
        if self.exc_eigenvectors is None:
            raise RuntimeError("Eigenvectors not loaded. Use load_eigenvectors=True.")

        w_qe = projwfc.get_weights(selected_orbitals=selected_orbitals)  # (nk, nbands)

        k_idx = self.exc_table[:, 0] - 1  # (ntransitions,) 0-based, full-BZ indices
        v_idx = self.exc_table[:, 1] - 1
        c_idx = self.exc_table[:, 2] - 1

        # If projwfc was run on the IBZ only, map full-BZ k-indices to IBZ.
        # Orbital projections are invariant under crystal symmetry operations.
        if w_qe.shape[0] == self.lattice.ibz_nkpoints and hasattr(self.lattice, 'kpoints_indexes'):
            k_idx = self.lattice.kpoints_indexes[k_idx]

        if contribution == 'valence':
            w_t = w_qe[k_idx, v_idx]
        elif contribution == 'conduction':
            w_t = w_qe[k_idx, c_idx]
        elif contribution == 'both':
            w_t = 0.5 * (w_qe[k_idx, v_idx] + w_qe[k_idx, c_idx])
        else:
            raise ValueError("contribution must be 'valence', 'conduction', or 'both'")

        A2 = np.abs(self.exc_eigenvectors) ** 2          # (nq, nexcitons, ntransitions)
        return np.einsum('qst,t->qs', A2, w_t)           # (nq, nexcitons)

    def plot_orbital_projected_dispersion(
        self, path, projwfc, orbital_groups,
        contribution='both', tol=1e-3, expand_bz=True,
        interpolate=False, method='cubic_spline', npts=300, lpratio=6, nelect=1,
        s=200, lw=1.5, line_color='black', ylim=None, figsize=(8, 5), title=None,
        alpha=0.7, **scatter_kw,
    ):
        """
        Exciton dispersion with marker size proportional to orbital character.

        Orbital weights are only shown on the computed Q-points (scatter).
        When interpolate=True, a thin line connects the energies to guide the eye
        but the line is NOT orbital-projected (interpolating weights is unreliable).

        Parameters
        ----------
        path          : Path object
        projwfc       : ProjwfcXML — QE orbital projections
        orbital_groups: list of dicts, each with:
                          'orbitals' : list of state indices (from get_states_helper)
                          'color'    : matplotlib color
                          'label'    : legend label
        contribution  : 'valence' | 'conduction' | 'both' (default)
        tol           : Q-point path tolerance in Ang^-1
        expand_bz     : include symmetry-expanded Q-points (shown as triangles)
        interpolate   : if True, draw interpolated line beneath the scatter dots
        method        : interpolation method — 'cubic_spline' | 'rbf' | 'skw' | 'nn'
        npts          : number of dense points for the interpolated line
        lpratio, nelect : SKW parameters
        s             : base marker area (weight=1.0 → area=s)
        alpha         : marker transparency (default 0.7); pass 1.0 for fully opaque
        scatter_kw    : extra keyword arguments forwarded to ax.scatter
        lw            : line width for the interpolated line
        line_color    : color of the interpolated line
        ylim          : (ymin, ymax) energy window in eV
        figsize, title: figure layout

        Example
        -------
        groups = [
            {'orbitals': proj.get_states_helper(['Mo'], ['d']), 'color': 'red',  'label': 'Mo-d'},
            {'orbitals': proj.get_states_helper(['S'],  ['p']), 'color': 'blue', 'label': 'S-p'},
        ]
        fig, ax = exc_disp.plot_orbital_projected_dispersion(path, proj, groups,
                                                             interpolate=True)
        """
        fig, ax = plt.subplots(figsize=figsize)

        if interpolate:
            dense_x, dense_e, scatter_x, scatter_e, is_ibz, boundaries, labels = \
                self.get_dispersion_interpolated(
                    path, method=method, npts=npts, tol=tol, expand_bz=expand_bz,
                    lpratio=lpratio, nelect=nelect,
                )
            for ib in range(self.nexcitons):
                ax.plot(dense_x, dense_e[:, ib], color=line_color, lw=lw, zorder=1)
            _, q_indices, _, _, _ = self._project_qpts_onto_path(
                path, tol=tol, expand_bz=expand_bz
            )
        else:
            scatter_x, q_indices, is_ibz, boundaries, labels = \
                self._project_qpts_onto_path(path, tol=tol, expand_bz=expand_bz)
            energies_full = self.exc_energies[self.lattice.kpoints_indexes]
            scatter_e     = energies_full[q_indices]      # (N, nexcitons)

        for group in orbital_groups:
            orb_weights  = self.get_orbital_weights(
                projwfc, group['orbitals'], contribution=contribution
            )                                             # (nq_ibz, nexcitons)
            weights_full = orb_weights[self.lattice.kpoints_indexes]
            scatter_w    = weights_full[q_indices]        # (N, nexcitons)

            color = group.get('color', 'black')
            label = group.get('label', '')

            for ib in range(self.nexcitons):
                for mask, marker in [(is_ibz, 'o'), (~is_ibz, '^')]:
                    if not mask.any():
                        continue
                    lab = label if (ib == 0 and marker == 'o') else '_' + label
                    ax.scatter(
                        scatter_x[mask], scatter_e[mask, ib],
                        s=scatter_w[mask, ib] * s,
                        color=color, marker=marker, label=lab,
                        edgecolors='none', linewidths=0, zorder=3,
                        alpha=alpha, **scatter_kw,
                    )

        for x in boundaries:
            ax.axvline(x, color='gray', lw=1.5, ls='--')
        ax.set_xticks(boundaries)
        ax.set_xticklabels(labels)
        ax.set_xlim(boundaries[0], boundaries[-1])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("Exciton energy (eV)")
        if title is None:
            title = "Orbital-projected exciton dispersion (%s)" % contribution
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------

    def __str__(self):
        lines = []; app = lines.append
        app(" Exciton Dispersion")
        app(" Q-points (IBZ):              %d" % self.nqpoints)
        app(" BSE basis size (nk*nv*nc):   %d" % self.ntransitions)
        app(" Exciton states available:     %d" % self.nexcitons_available)
        app(" Exciton states loaded:        %d" % self.nexcitons)
        return "\n".join(lines)
