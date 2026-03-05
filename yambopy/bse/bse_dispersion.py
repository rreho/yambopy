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

    def _sample_path_cartesian(self, path, path_npoints):
        """
        Sample a BZ path at regular intervals.
        Returns sampled Cartesian coords, cumulative distances, and segment boundary distances.
        """
        kpts_path_car      = red_car(path.kpoints, self.rlat)
        sampled_car        = []
        sampled_kpath      = []
        boundary_distances = [0.0]
        cumulative         = 0.0

        for k in range(len(path.kpoints) - 1):
            start   = kpts_path_car[k]
            end     = kpts_path_car[k + 1]
            seg_len = np.linalg.norm(end - start)
            for i in range(path_npoints):
                t = i / path_npoints
                sampled_car.append(start + t * (end - start))
                sampled_kpath.append(cumulative + t * seg_len)
            cumulative += seg_len
            boundary_distances.append(cumulative)

        sampled_car.append(kpts_path_car[-1])
        sampled_kpath.append(cumulative)

        return np.array(sampled_car), np.array(sampled_kpath), np.array(boundary_distances)


    def _get_full_bz_qpoints(self):
        """
        Replicate IBZ q-mesh over neighbouring BZ images and convert to Cartesian.
        Returns (qpoints_rep, qpoints_idx_rep, car_qpoints).
        """
        rep = list(range(-1, 2))
        qpoints_rep, qpoints_idx_rep = replicate_red_kmesh(
            self.red_qpoints, repx=rep, repy=rep, repz=[0]
        )
        car_qpoints = red_car(qpoints_rep, self.rlat)
        return qpoints_rep, qpoints_idx_rep, car_qpoints


    def _expand_ibz_to_full_bz(self, data_ibz):
        """
        Expand a scalar quantity from IBZ to full BZ using kpoints_indexes.
        data_ibz: (nq_ibz, ...) -> data_full: (nq_full, ...)
        For scalars (energy) this is just indexing.
        """
        return data_ibz[self.lattice.kpoints_indexes]


    def _compute_spin_full_bz(self, nstates, save_dir, bse_dir, contribution):
        """
        Compute S_z expectation values for all full BZ q-points directly.
        Returns spin_full of shape (nq_full, nstates).
        """
        from yambopy.bse.exciton_spin import compute_exc_spin_iqpt
        nq_full  = len(self.lattice.kpoints_indexes)
        spin_full = np.zeros((nq_full, nstates))

        for iq in range(nq_full):
            exe_Sz, _ = compute_exc_spin_iqpt(
                path=save_dir, bse_dir=bse_dir,
                iqpt=iq + 1,
                nstates=nstates, contribution=contribution,
                return_dbs_and_spin=True
            )
            spin_full[iq] = exe_Sz[:nstates]

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

    def get_dispersion(self, path, path_npoints=50):
        """
        Nearest-neighbour q-point matching along path. No interpolation.
        Returns: bands, distances, boundaries, labels, exc_indexes
        """
        _, qpoints_idx_rep, car_qpoints = self._get_full_bz_qpoints()
        sampled_car, sampled_kpath, boundaries = self._sample_path_cartesian(path, path_npoints)

        from scipy.spatial import cKDTree
        _, nn_indices  = cKDTree(car_qpoints).query(sampled_car, k=1)
        exc_indexes    = qpoints_idx_rep[nn_indices]

        unique, counts = np.unique(exc_indexes, return_counts=True)
        print(f"Unique q-points on path: {len(unique)} / {len(self.red_qpoints)}")

        energies_full = self._expand_ibz_to_full_bz(self.exc_energies)
        bands         = energies_full[exc_indexes]

        return bands, sampled_kpath, boundaries, path.klabels, exc_indexes

    def get_dispersion_interpolated(self, path, path_npoints=50,
                                    show_spin=False, save_dir='SAVE',
                                    bse_dir='BSE', contribution='b'):

        _, qpoints_idx_rep, car_qpoints = self._get_full_bz_qpoints()
        sampled_car, sampled_kpath, boundaries = self._sample_path_cartesian(path, path_npoints)

        nstates     = self.exc_energies.shape[1]
        eigens_full = self._expand_ibz_to_full_bz(self.exc_energies)

        if show_spin:
            spin_full  = self._compute_spin_full_bz(nstates, save_dir, bse_dir, contribution)
            bands, spin_path = self._nn_interpolate(sampled_car, car_qpoints,
                                                    qpoints_idx_rep,
                                                    eigens_full, spin_full)
        else:
            bands, = self._nn_interpolate(sampled_car, car_qpoints,
                                        qpoints_idx_rep, eigens_full)
            spin_path = None

        return bands, sampled_kpath, boundaries, path.klabels, spin_path
   
    def get_spin_along_path(self, exc_indexes, nstates,
                            save_dir='SAVE', bse_dir='BSE', contribution='b'):
        from yambopy.bse.exciton_spin import compute_exc_spin_iqpt
        spin  = np.zeros((len(exc_indexes), nstates))
        cache = {}

        for i, iq_full in enumerate(exc_indexes):
            iq_fortran = int(iq_full) + 1
            if iq_fortran not in cache:
                exe_Sz, _ = compute_exc_spin_iqpt(
                    path=save_dir, bse_dir=bse_dir,
                    iqpt=iq_fortran, nstates=nstates,
                    contribution=contribution, degen_tol=1e-4,
                    return_dbs_and_spin=True
                )
                cache[iq_fortran] = exe_Sz[:nstates]
            spin[i] = cache[iq_fortran]

        return spin


    def plot_exciton_dispersion(self, path, ylim=None, figsize=(8, 5),
                                title="Exciton dispersion", save_dir='SAVE', bse_dir='BSE',
                                contribution='b', show_spin=False, interpolate=False):
        """
        Plot exciton dispersion along a BZ path.
        - interpolate=False: snap to nearest q-point (fast, steppy for coarse mesh)
        - interpolate=True:  linear interpolation between 2 nearest q-points (smoother)
        - show_spin=True:    color bands by S_z expectation value
        """
        if interpolate:
            bands, distances, boundaries, labels, spin = self.get_dispersion_interpolated(
                path=path, show_spin=show_spin,
                save_dir=save_dir, bse_dir=bse_dir, contribution=contribution
            )
        else:
            bands, distances, boundaries, labels, exc_indices = self.get_dispersion(path=path)
            spin = self.get_spin_along_path(exc_indices, bands.shape[1],
                                            save_dir=save_dir, bse_dir=bse_dir,
                                            contribution=contribution) if show_spin else None

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
