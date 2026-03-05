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
from yambopy.plot.plotting import add_fig_kwargs,BZ_Wigner_Seitz
from yambopy.lattice import replicate_red_kmesh, calculate_distances, car_red
from yambopy.kpoints import get_path
from yambopy.tools.funcs import gaussian, lorentzian
import matplotlib.pyplot as plt
import numpy as np

class ExcitonDispersion():
    """
    Class to obtain exciton information at all momenta

    - Dispersion plot (under development)
    - Plots of exciton weights in q-space

    :: Lattice is an instance of YamboLatticeDB
    :: nexcitons is the number of excitonic states - by default it is taken from the Q=1 database

    NB: so far does not support spin-polarised exciton plots (should be implemented when needed!)
    NB: only supports BSEBands option in yambo bse input, not BSEEhEny
    """

    def __init__(self,lattice,nexcitons=None,folder='.'):

        if not isinstance(lattice,YamboLatticeDB):
            raise ValueError('Invalid type for lattice argument. It must be YamboLatticeDB')
        
        files   = glob(folder+'/ndb.BS_diago_Q*')
        nqpoints  = len(files)

        # Check
        if not nqpoints==lattice.ibz_nkpoints :
            raise ValueError("Incomplete list of qpoints (%d/%d)"%(nqpoints,lattice.ibz_nkpoints)) 
    
        dbs_are_consistent, spin_is_there = self.db_check(lattice,nqpoints,folder)
        if nexcitons is None: nexcitons = self.ntransitions

        # Read
        car_qpoints         = np.zeros((nqpoints,3))
        exc_energies        = np.zeros((nqpoints,nexcitons))
        exc_eigenvectors    = np.zeros((nqpoints,nexcitons,self.ntransitions),dtype=complex)
        exc_tables          = np.zeros((nqpoints,self.ntransitions,5),dtype=int)
        for iQ in range(nqpoints):
            exc_obj = YamboExcitonDB.from_db_file(lattice,filename=folder+'/ndb.BS_diago_Q%d'%(iQ+1))
            if iQ==0: car_qpoints[iQ] = np.array([0.,0.,0.])
            else:     car_qpoints[iQ] = exc_obj.car_qpoint
            exc_energies[iQ,:]        = exc_obj.eigenvalues[:nexcitons].real
            exc_eigenvectors[iQ,:]    = exc_obj.eigenvectors[:nexcitons]    
            exc_tables[iQ,:]          = exc_obj.table

        # Set up variables
        self.folder = folder
        self.nqpoints     = nqpoints
        self.nexcitons    = nexcitons
        self.car_qpoints  = car_qpoints
        self.red_qpoints  = car_red(car_qpoints,lattice.rlat)
        self.lattice      = lattice
        self.exc_energies = exc_energies
        self.exc_tables   = exc_tables

        # Reshape eigenvectors if possible
        if dbs_are_consistent and not spin_is_there: self.exc_eigenvectors = self.reshape_eigenvectors(exc_eigenvectors)
        else:                                        self.exc_eigenvectors = exc_eigenvectors

        # Necessary lattice information
        self.alat = lattice.alat
        self.rlat = lattice.rlat

    def db_check(self,lattice,nqpoints,folder):
        """
        Check nexcitons and ntransitions in each database
        """
        nexcitons_each_Q    = np.zeros(nqpoints,dtype=int)
        for iQ in range(nqpoints):
            exc_obj = YamboExcitonDB.from_db_file(lattice,filename=folder+'/ndb.BS_diago_Q%d'%(iQ+1))
            nexcitons_each_Q[iQ] = exc_obj.nexcitons
            if iQ==0: tbl = exc_obj.table

        is_spin_pol = len(np.unique(tbl[:,3]))>1 or len(np.unique(tbl[:,4]))>1
        is_consistent = np.all(nexcitons_each_Q==nexcitons_each_Q[0])
        if not is_consistent: 
            print("[WARNING] BSE Hamiltonian has different dimensions for some Q.")
            print("          Taking the minimum number of transitions, be careful.")
            self.ntransitions = np.min(nexcitons_each_Q)
        else:
            if is_spin_pol: print("[WARNING] Spin-polarised excitons, only partially supported")
            self.ntransitions = nexcitons_each_Q[0]
            valence_bands    = np.unique(tbl[:,1]) - 1
            conduction_bands = np.unique(tbl[:,2]) - 1
            self.nkpoints    = np.max(tbl[:,0])
            self.nvalence    = len(valence_bands)
            self.nconduction = len(conduction_bands)

        return is_consistent, is_spin_pol

    def reshape_eigenvectors(self,eigenvectors):
        """
        eigenvectors in:  [nqpoints, nexcitons, ntransitions]
        eigenvectors out: [nqpoints, nexcitons, nkpoints, nvalence, nconduction]

        TODO: Extend to spin-polarised case
        """
        nq, nexc, nk, nv, nc = self.nqpoints, self.nexcitons, self.nkpoints, self.nvalence, self.nconduction
        reshaped_eigenvectors = np.zeros((nq,nexc,nk,nv,nc),dtype=complex)
        #print(eigenvectors[2,5,2])
        for iQ in range(nq):
            for i_exc in range(nexc): 
                reshaped_eigenvectors[iQ,i_exc,:,:,:] = eigenvectors[iQ,i_exc,:].reshape([nk,nv,nc])
        #print(reshaped_eigenvectors[2,5,0,1,0])    

        return reshaped_eigenvectors

    @add_fig_kwargs
    def plot_Aweights(self,data,plt_show=False,plt_cbar=False,**kwargs):
        """
        2D scatterplot in the q-BZ of the quantity A_{iq}(iexc,ik,ic,iv).

        Any real quantity which is a function of only the q-grid may be supplied.
        The indices iq,inu,ib1,ib2 are user-specified.

        - if plt_show plot is shown
        - if plt_cbar colorbar is shown
        - kwargs example: marker='H', s=300, cmap='viridis', etc.

        NB: So far requires a 2D system.
            Can be improved to plot BZ planes at constant k_z for 3D systems.
        """

        qpts = self.car_qpoints

        # Input check
        if len(data)!=len(qpts):
            raise ValueError('Something wrong in data dimensions (%d data vs %d qpts)'%(len(data),len(qpts)))

        # Global plot stuff
        self.fig, self.ax = plt.subplots(1, 1)
        if self.nqpoints<self.nkpoints:  c_BZ_borders='black'
        if self.nqpoints==self.nkpoints: c_BZ_borders='white'
        self.ax.add_patch(BZ_Wigner_Seitz(self.lattice,color=c_BZ_borders))


        if plt_cbar:
            if 'cmap' in kwargs.keys(): color_map = plt.get_cmap(kwargs['cmap'])
            else:                       color_map = plt.get_cmap('viridis')
        lim = 1.05*np.linalg.norm(self.rlat[0])
        self.ax.set_xlim(-lim,lim)
        self.ax.set_ylim(-lim,lim)

        # Reproduce plot also in adjacent BZs
        BZs = shifted_grids_2D(qpts,self.rlat)
        for qpts_s in BZs: plot=self.ax.scatter(qpts_s[:,0],qpts_s[:,1],c=data,**kwargs)

        if plt_cbar: self.fig.colorbar(plot)

        plt.gca().set_aspect('equal')

        if plt_show: plt.show()
        else: print("Plot ready.\nYou can customise adding savefig, title, labels, text, show, etc...")
    #####################################
    # Excition dispersion along BZ path #
    #####################################
    def _sample_path_cartesian(self, path):
        """
        Sample a BZ path using path.get_klist() which respects path.intervals.
        Returns sampled Cartesian coords, cumulative distances, and segment boundary distances.
        """
        klist = path.get_klist()                    # (npath, 4) reduced coords + weight
        red_kpts = klist[:, :3]                     # (npath, 3)
        car_kpts = red_car(red_kpts, self.rlat)     # (npath, 3)

        # Compute cumulative distances in Cartesian
        diffs      = np.linalg.norm(np.diff(car_kpts, axis=0), axis=1)
        cumulative = np.concatenate([[0], np.cumsum(diffs)])

        # Boundary distances at high-symmetry points
        boundary_distances = [0.0]
        idx = 0
        for npts in path.intervals:
            idx += npts
            boundary_distances.append(cumulative[idx])

        return car_kpts, cumulative, np.array(boundary_distances)
    
    def _get_full_bz_qpoints(self):
        """
        Return full BZ q-points with BZ images to handle folding.
        """
        red_kpoints = self.lattice.red_kpoints          # (nq_full, 3)
        nq          = len(red_kpoints)

        # Add BZ images so path points near zone boundary find correct q-point
        shifts = np.array([[i, j, 0] for i in [-1,0,1] for j in [-1,0,1]])
        red_rep = np.vstack([red_kpoints + s for s in shifts])   # (9*nq, 3)
        idx_rep = np.tile(np.arange(nq), len(shifts))            # maps back to original

        car_rep = red_car(red_rep, self.rlat)
        return red_kpoints, idx_rep, car_rep


    def _expand_ibz_to_full_bz(self, data_ibz):
        """
        Expand a scalar quantity from IBZ to full BZ using kpoints_indexes.
        data_ibz: (nq_ibz, ...) -> data_full: (nq_full, ...)
        For scalars (energy) this is just indexing.
        """
        return data_ibz[self.lattice.kpoints_indexes]
    
    def _compute_spin_ibz(self, nstates, save_dir, bse_dir, contribution,
                        sz=0.5 * np.array([[1, 0], [0, -1]])):
        """
        Compute S_z expectation values for all IBZ q-points.
        Loads wfdb and elec_sz once, then loops over IBZ q-points.
        Returns spin_ibz of shape (nq_ibz, nstates).
        """
        from yambopy.bse.exciton_spin import compute_exciton_spin, get_spinvals
        from yambopy.dbs.wfdb import YamboWFDB

        nq_ibz   = len(self.red_qpoints)
        spin_ibz = np.zeros((nq_ibz, nstates))

        # Load excdb once just to get bands_range
        excdb_q1 = YamboExcitonDB.from_db_file(
            self.lattice,
            filename='ndb.BS_diago_Q1',
            folder=bse_dir,
            Load_WF=True, neigs=nstates
        )
        bands_range = [np.min(excdb_q1.table[:, 1]) - 1,
                    np.max(excdb_q1.table[:, 2])]

        # Load wfdb and elec_sz once — these don't depend on q-point
        wfdb    = YamboWFDB(path=save_dir, latdb=self.lattice, bands_range=bands_range)
        elec_sz = wfdb.get_spin_m_e_BZ(s_z=sz)

        # Now loop over IBZ q-points, reusing wfdb and elec_sz
        for iq in range(nq_ibz):
            if iq == 0:
                excdb = excdb_q1   # already loaded
            else:
                excdb = YamboExcitonDB.from_db_file(
                    self.lattice,
                    filename=f'ndb.BS_diago_Q{iq + 1}',
                    folder=bse_dir,
                    Load_WF=True, neigs=nstates
                )

            smat   = compute_exciton_spin(self.lattice, excdb, wfdb, elec_sz,
                                        contribution=contribution, diagonal=False)
            smat   = get_spinvals(smat, excdb.eigenvalues, atol=1e-2)

            ss_tmp = []
            for i in smat:
                ss_tmp += list(i)
            spin_ibz[iq] = np.array(ss_tmp)[:nstates].real

        return spin_ibz
    def _compute_spin_full_bz(self, nstates, save_dir, bse_dir, contribution,
                           sz=0.5 * np.array([[1, 0], [0, -1]]),
                           dmat_mode='run', dmat_file='Dmats.npy',
                           method='rotate_Ak'):
        """
        Compute S_z for full BZ.
        method='Rzz'        : apply Rzz symmetry transformation to IBZ spin (fast)
        method='rotate_Ak'  : rotate exciton wavefunction to full BZ q-point (slow)
        """
        from yambopy.bse.exciton_spin import compute_exciton_spin, get_spinvals
        from yambopy.dbs.wfdb import YamboWFDB

        nq_ibz  = len(self.red_qpoints)
        nq_full = len(self.lattice.kpoints_indexes)

        # --- Load wfdb and elec_sz once ---
        excdb_q1 = YamboExcitonDB.from_db_file(
            self.lattice, filename='ndb.BS_diago_Q1',
            folder=bse_dir, Load_WF=True, neigs=nstates
        )
        bands_range = [np.min(excdb_q1.table[:, 1]) - 1,
                    np.max(excdb_q1.table[:, 2])]
        wfdb    = YamboWFDB(path=save_dir, latdb=self.lattice, bands_range=bands_range)
        elec_sz = wfdb.get_spin_m_e_BZ(s_z=sz)

        # --- Load all IBZ exciton dbs ---
        exdbs = [excdb_q1]
        for iq in range(1, nq_ibz):
            exdbs.append(YamboExcitonDB.from_db_file(
                self.lattice, filename=f'ndb.BS_diago_Q{iq + 1}',
                folder=bse_dir, Load_WF=True, neigs=nstates
            ))

        # --- Compute spin at IBZ q-points ---
        spin_ibz = np.zeros((nq_ibz, nstates))
        for iq, excdb in enumerate(exdbs):
            smat = compute_exciton_spin(self.lattice, excdb, wfdb, elec_sz,
                                        contribution=contribution, diagonal=False)
            smat = get_spinvals(smat, excdb.eigenvalues, atol=1e-2)
            ss_tmp = []
            for i in smat:
                ss_tmp += list(i)
            spin_ibz[iq] = np.array(ss_tmp)[:nstates].real

        # --- Expand to full BZ ---
        spin_full = np.zeros((nq_full, nstates))

        if method == 'Rzz':
            for iq_full, (iq_ibz, isym) in enumerate(zip(self.lattice.kpoints_indexes,
                                                        self.lattice.symmetry_indexes)):
                Rzz = self.lattice.sym_red[isym][2, 2]
                spin_full[iq_full] = Rzz * spin_ibz[iq_ibz]

        elif method == 'rotate_Ak':
            from yambopy.exciton_phonon.excph_matrix_elements import rotate_Akcv_Q, save_or_load_dmat
            from scipy.spatial import cKDTree

            Dmats = save_or_load_dmat(wfdb, mode=dmat_mode, dmat_file=dmat_file)

            for iq_full in range(nq_full):
                Qpt    = self.lattice.red_kpoints[iq_full]
                rot_Ak = rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats, folder=None)

                iq_ibz          = self.lattice.kpoints_indexes[iq_full]
                excdb           = exdbs[iq_ibz]
                original_get_Akcv = excdb.get_Akcv
                excdb.get_Akcv  = lambda: rot_Ak

                smat = compute_exciton_spin(self.lattice, excdb, wfdb, elec_sz,
                                            contribution=contribution, diagonal=False)
                smat = get_spinvals(smat, excdb.eigenvalues, atol=1e-2)

                excdb.get_Akcv = original_get_Akcv

                ss_tmp = []
                for i in smat:
                    ss_tmp += list(i)
                spin_full[iq_full] = np.array(ss_tmp)[:nstates].real

        else:
            raise ValueError(f"Unknown method '{method}', use 'Rzz' or 'rotate_Ak'")

        return spin_full
    
    def _nn_interpolate(self, sampled_car, car_qpoints, qpoints_idx_rep, *data_full):
        """
        Linear interpolation using 2 nearest neighbours in Cartesian space.
        qpoints_idx_rep maps replicated mesh indices back to full BZ indices.
        """
        from scipy.spatial import cKDTree
        dists, nn_indices = cKDTree(car_qpoints).query(sampled_car, k=2)

        d0, d1 = dists[:, 0], dists[:, 1]
        total  = d0 + d1
        exact  = total < 1e-10
        w0 = np.where(exact, 1.0, d1 / total)[:, np.newaxis]
        w1 = np.where(exact, 0.0, d0 / total)[:, np.newaxis]

        # Map replicated indices back to full BZ indices
        idx0 = qpoints_idx_rep[nn_indices[:, 0]]
        idx1 = qpoints_idx_rep[nn_indices[:, 1]]

        return [w0 * d[idx0] + w1 * d[idx1] for d in data_full]

    def get_dispersion(self, path):
        """
        Nearest-neighbour q-point matching along path.
        Uses path.intervals for npoints per segment if path_npoints is None.
        """
        _, qpoints_idx_rep, car_qpoints = self._get_full_bz_qpoints()
        sampled_car, sampled_kpath, boundaries = self._sample_path_cartesian(path)
        
        from scipy.spatial import cKDTree
        _, nn_indices  = cKDTree(car_qpoints).query(sampled_car, k=1)
        exc_indexes    = qpoints_idx_rep[nn_indices]

        unique, counts = np.unique(exc_indexes, return_counts=True)
        print(f"Unique q-points on path: {len(unique)} / {len(self.red_qpoints)}")

        energies_full = self._expand_ibz_to_full_bz(self.exc_energies)
        bands         = energies_full[exc_indexes]

        return bands, sampled_kpath, boundaries, path.klabels, exc_indexes
    
    def _interpolate_rbf(self, sampled_car, eigens, eigens_full_bz=None,
                        enforce_symmetry_points=None):
        from scipy.interpolate import RBFInterpolator

        # Fold full BZ q-points into [-0.5, 0.5) to ensure consistent representation


        car_qpoints_full   = red_car(self.lattice.red_kpoints, self.rlat)

        if eigens_full_bz is not None:
            eigens_full = eigens_full_bz
        else:
            eigens_full = self._expand_ibz_to_full_bz(eigens)

        # Also fold the sampled path points
        sampled_red    = car_red(sampled_car, self.rlat)
        sampled_folded = (sampled_red + 0.5) % 1.0 - 0.5
        sampled_car_folded = red_car(sampled_folded, self.rlat)

        pts_2d    = car_qpoints_full[:, :2]
        sample_2d = sampled_car_folded[:, :2]
        nbands    = eigens_full.shape[1]
        result    = np.zeros((len(sampled_car), nbands))

        kernels = ['linear', 'thin_plate_spline', 'multiquadric', 'gaussian']
        for kernel in kernels:
            try:
                for ib in range(nbands):
                    rbf = RBFInterpolator(
                        pts_2d, eigens_full[:, ib],
                        kernel=kernel, smoothing=0.0, degree=-1
                    )
                    result[:, ib] = rbf(sample_2d)
                print(f"RBF succeeded with kernel='{kernel}'")
                return result
            except Exception as e:
                print(f"RBF kernel '{kernel}' failed ({e}), trying next...")

        raise RuntimeError("All RBF kernels failed")
    
    def get_dispersion_interpolated(self, path, show_spin=False,
                                    save_dir='SAVE', bse_dir='BSE', contribution='b',
                                    dmat_mode='run', dmat_file='Dmats.npy',
                                    use_skw=True, lpratio=5, spin_method='rotate_Ak'):

        _, qpoints_idx_rep, car_qpoints = self._get_full_bz_qpoints()
        sampled_car, sampled_kpath, boundaries = self._sample_path_cartesian(path)

        nstates     = self.exc_energies.shape[1]
        eigens_full = self._expand_ibz_to_full_bz(self.exc_energies)
        spin_path   = None

        # --- Interpolate energies ---
        bands = None
        if use_skw:
            try:
                from yambopy.tools.skw import SkwInterpolator
                sampled_red = car_red(sampled_car, self.rlat)
                skw = SkwInterpolator(
                    lpratio    = lpratio,
                    kpts       = self.red_qpoints,
                    eigens     = self.exc_energies[np.newaxis, :, :],
                    fermie     = 0.0, nelect = 0,
                    cell       = (self.lattice.lat,
                                self.lattice.red_atomic_positions,
                                self.lattice.atomic_numbers),
                    symrel     = self.lattice.sym_red,
                    has_timrev = bool(self.lattice.time_rev),
                    verbose    = 1
                )
                if skw.mae > 10.0:
                    raise ValueError(f"SKW MAE too large ({skw.mae:.1f} meV)")
                bands = skw.interp_kpts(sampled_red).eigens[0]
                print(f"SKW interpolation succeeded, MAE={skw.mae:.3f} meV")

            except Exception as e:
                print(f"SKW failed ({e}), trying RBF...")
                try:
                    car_qpoints_ibz = red_car(self.red_qpoints, self.rlat)
                    bands = self._interpolate_rbf(sampled_car, self.exc_energies)                    
                    print("RBF interpolation succeeded")
                except Exception as e2:
                    print(f"RBF failed ({e2}), falling back to NN")

        # NN fallback (also used when use_skw=False)
        if bands is None:
            if show_spin:
                spin_full = self._compute_spin_full_bz(nstates, save_dir, bse_dir,
                                                        contribution, dmat_mode=dmat_mode,
                                                        dmat_file=dmat_file, method=spin_method)
                bands, spin_path = self._nn_interpolate(sampled_car, car_qpoints,
                                                        qpoints_idx_rep,
                                                        eigens_full, spin_full)
            else:
                bands, = self._nn_interpolate(sampled_car, car_qpoints,
                                            qpoints_idx_rep, eigens_full)
            return bands, sampled_kpath, boundaries, path.klabels, spin_path

        # --- Interpolate spin (only if bands came from SKW or RBF) ---
        if show_spin:
            spin_full = self._compute_spin_full_bz(nstates, save_dir, bse_dir,
                                                    contribution, dmat_mode=dmat_mode,
                                                    dmat_file=dmat_file, method=spin_method)
            try:
                # Try RBF for spin (SKW not used since spin is not periodic in same way)
                car_qpoints_full = red_car(self.lattice.red_kpoints, self.rlat)
                spin_path = self._interpolate_rbf(sampled_car, None, eigens_full_bz=spin_full)
                print("RBF spin interpolation succeeded")
            except Exception as e:
                print(f"RBF spin failed ({e}), falling back to NN spin")
                _, spin_path = self._nn_interpolate(sampled_car, car_qpoints,
                                                    qpoints_idx_rep,
                                                    eigens_full, spin_full)

        return bands, sampled_kpath, boundaries, path.klabels, spin_path
        
    def _expand_spin_to_full_bz(self, spin_ibz):
        """
        Expand S_z from IBZ to full BZ applying symmetry transformations.
        S_z transforms as a pseudovector: S_z -> sym_red[isym][2,2] * S_z
        """
        sym_red   = self.lattice.sym_red
        nq_full   = len(self.lattice.kpoints_indexes)
        spin_full = np.zeros((nq_full, spin_ibz.shape[1]))

        for iq_full, (iq_ibz, isym) in enumerate(zip(self.lattice.kpoints_indexes,
                                                    self.lattice.symmetry_indexes)):
            Rzz = sym_red[isym][2, 2]
            spin_full[iq_full] = Rzz * spin_ibz[iq_ibz]

        return spin_full


    def get_spin_along_path(self, exc_indexes_full, nstates,
                            save_dir='SAVE', bse_dir='BSE', contribution='b',
                            dmat_mode='save', dmat_file='Dmats.npy', spin_method='rotate_Ak'):
        spin_full = self._compute_spin_full_bz(nstates, save_dir, bse_dir, contribution,
                                                dmat_mode=dmat_mode, dmat_file=dmat_file, method=spin_method)
        return spin_full[exc_indexes_full]

    def plot_exciton_dispersion(self, path, ylim=None, figsize=(8, 5),
                                title="Exciton dispersion", save_dir='SAVE', bse_dir='BSE',
                                contribution='b', show_spin=False, interpolate=False,
                                dmat_mode='save', dmat_file='Dmats.npy',spin_method='rotate_Ak'):
        if interpolate:
            bands, distances, boundaries, labels, spin = self.get_dispersion_interpolated(
                path=path, show_spin=show_spin, save_dir=save_dir, bse_dir=bse_dir,
                contribution=contribution, dmat_mode=dmat_mode, dmat_file=dmat_file, spin_method=spin_method
            )
        else:
            bands, distances, boundaries, labels, exc_indexes_full = self.get_dispersion(path=path)
            spin = self.get_spin_along_path(
                exc_indexes_full, bands.shape[1], save_dir=save_dir, bse_dir=bse_dir,
                contribution=contribution, dmat_mode=dmat_mode, dmat_file=dmat_file, spin_method=spin_method
            ) if show_spin else None

        fig, ax = plt.subplots(figsize=figsize)

        if show_spin and spin is not None:
            norm = plt.Normalize(vmin=-0.5, vmax=0.5)
            cmap = plt.cm.RdBu
            for ib in range(bands.shape[1]):
                sc = ax.scatter(distances, bands[:, ib], c=spin[:, ib],
                                cmap=cmap, norm=norm, s=8, linewidths=0,
                                zorder=2, label=f"Exciton {ib+1}")
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$\langle S_z \rangle$")
            cbar.set_ticks([-0.5, 0, 0.5])
        else:
            colors = plt.cm.viridis(np.linspace(0, 0.85, bands.shape[1]))
            for ib in range(bands.shape[1]):
                ax.plot(distances, bands[:, ib], color=colors[ib], lw=1.5,
                        label=f"Exciton {ib+1}")
            ax.legend(fontsize=8)

        for x in boundaries:
            ax.axvline(x, color='gray', lw=0.8, ls='--')
        ax.set_xticks(boundaries)
        ax.set_xticklabels(labels)
        ax.set_xlim(distances[0], distances[-1])
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel("Exciton energy (eV)")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    def plot_exciton_disp_ax(self,ax,path,**kwargs):
        ybs_disp = self.get_dispersion(path)
        print(ybs_disp.nbands)
        print(ybs_disp.nkpoints)
        print(ybs_disp._xlim)
        return ybs_disp.plot_ax(ax) 

    @add_fig_kwargs
    def plot_exciton_disp(self,path,**kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_exciton_disp_ax(ax,path)
        return fig

    
    def __str__(self):
        lines = []; app = lines.append
        app(" Exciton Dispersion ")
        app(" Number of qpoints:                    %d"%self.nqpoints)
        app(" Number of exciton branches read:      %d"%self.nexcitons)
        app(" Total number of excitons/transitions: %d"%self.ntransitions)
        return "\n".join(lines)
