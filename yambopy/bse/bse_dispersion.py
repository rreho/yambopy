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
    # Dispersion plot under development #
    #####################################
    def get_dispersion(self, path, path_npoints=50, atol=None):
        
        qpoints = self.red_qpoints
        rep = list(range(-1, 2))

        qpoints_rep, qpoints_idx_rep = replicate_red_kmesh(
            qpoints, repx=rep, repy=rep, repz=[0]
        )
        car_qpoints = red_car(qpoints_rep, lat=self.lattice.rlat)
        kpts_path_car = red_car(path.kpoints, self.rlat)

        sampled_car   = []
        sampled_kpath = []
        boundary_distances = [0.0]
        cumulative = 0.0

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
        sampled_car   = np.array(sampled_car)
        sampled_kpath = np.array(sampled_kpath)

        from scipy.spatial import cKDTree
        tree = cKDTree(car_qpoints)
        _, nn_indices = tree.query(sampled_car, k=1)
        exc_indexes = qpoints_idx_rep[nn_indices]   # 0-based index into self.red_qpoints

        unique, counts = np.unique(exc_indexes, return_counts=True)
        print(f"Unique q-points matched to path: {len(unique)} / {len(self.red_qpoints)}")
        print(f"Each used this many times (npath points snapped to it):")
        for iq, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  q-index {iq}: {c} path points")

        energies      = self.exc_energies[self.lattice.kpoints_indexes]
        energies_path = energies[exc_indexes]       # (npath, nbands)

        return energies_path, sampled_kpath, np.array(boundary_distances), path.klabels, exc_indexes


    def get_spin_along_path(self, exc_indexes, nstates,save_dir='SAVE', bse_dir='BSE', contribution='b'):
        """
        Compute Sz for each unique q-point along the path.
        Returns spin array of shape (npath, nstates).
        """
        npath   = len(exc_indexes)
        spin    = np.zeros((npath, nstates))

        # Cache results per unique q-point to avoid recomputing
        from yambopy.bse.exciton_spin import compute_exc_spin_iqpt
        cache = {}
        for i, iq0 in enumerate(exc_indexes):
            iq_fortran = int(iq0) + 1   # 0-based -> Fortran 1-based
            if iq_fortran not in cache:
                exe_Sz, _ = compute_exc_spin_iqpt(
                    path=save_dir,
                    bse_dir=bse_dir,
                    iqpt=iq_fortran,
                    nstates=nstates,
                    contribution=contribution,
                    return_dbs_and_spin=True
                )
                cache[iq_fortran] = exe_Sz[:nstates]   # (nstates,)
            spin[i] = cache[iq_fortran]

        return spin   # (npath, nstates)

    def get_dispersion_interpolated(self, path, path_npoints=50, lpratio=5):
        """
        Use SKW Fourier interpolation to get smooth exciton dispersion along path.
        """
        # --- Inputs for SKW ---
        # self.lattice.expand_kpoints()
        kpts      = self.lattice.red_kpoints                          # (nq, 3) reduced coords

        eigens    = self.exc_energies[self.lattice.kpoints_indexes]  # (nq, nbands)

        # SKW expects shape (nsppol, nkpt, nband)
        eigens_skw = eigens[np.newaxis, :, :]                 # (1, nq, nbands)

        # You need symmetry operations in real space (reduced coords)
        # If your lattice object has them:
        symrel   = self.lattice.sym_red                       # (nsym, 3, 3) — adjust attr name
        has_timrev = True

        cell = (self.lattice.lat,                             # real-space lattice vectors
                self.lattice.red_atomic_positions,                # reduced atomic positions
                self.lattice.atomic_numbers)                  # atomic numbers

        skw = SkwInterpolator(
            lpratio    = lpratio,
            kpts       = kpts,
            eigens     = eigens_skw,
            fermie     = 0.0,
            nelect     = 0,
            cell       = cell,
            symrel     = symrel,
            has_timrev = has_timrev,
            verbose    = 1
        )

        # --- Sample the path in reduced coords ---
        sampled_red   = []
        sampled_kpath = []
        boundary_distances = [0.0]
        cumulative = 0.0

        kpts_path_car = red_car(path.kpoints, self.rlat)

        for k in range(len(path.kpoints) - 1):
            start_red = path.kpoints[k]
            end_red   = path.kpoints[k + 1]
            start_car = kpts_path_car[k]
            end_car   = kpts_path_car[k + 1]
            seg_len   = np.linalg.norm(end_car - start_car)

            for i in range(path_npoints):
                t = i / path_npoints
                sampled_red.append(start_red + t * (end_red - start_red))
                sampled_kpath.append(cumulative + t * seg_len)

            cumulative += seg_len
            boundary_distances.append(cumulative)

        sampled_red.append(path.kpoints[-1])
        sampled_kpath.append(cumulative)
        sampled_red   = np.array(sampled_red)
        sampled_kpath = np.array(sampled_kpath)

        # --- Interpolate ---
        result    = skw.interp_kpts(sampled_red)
        bands_interp = result.eigens[0]              # (npath, nbands)

        return bands_interp, sampled_kpath, np.array(boundary_distances), path.klabels
    
    def plot_exciton_dispersion(self, path, ylim=None, figsize=(8, 5),
                                title="Exciton dispersion", save_dir='SAVE', bse_dir='BSE',
                                contribution='b', show_spin=False, interpolate=False):

        if interpolate:
            if show_spin:
                raise ValueError("show_spin=True requires interpolate=False (spin needs exact q-point indices)")
            bands, distances, boundaries, labels = self.get_dispersion_interpolated(path=path)
            exc_indices = None
        else:
            bands, distances, boundaries, labels, exc_indices = self.get_dispersion(path=path)
        
        nstates = bands.shape[1]
        fig, ax = plt.subplots(figsize=figsize)

        if show_spin:
            spin = self.get_spin_along_path(exc_indices, nstates,
                                            save_dir=save_dir, bse_dir=bse_dir,
                                            contribution=contribution)
            norm = plt.Normalize(vmin=-0.5, vmax=0.5)
            cmap = plt.cm.RdBu
            for ib in range(nstates):
                sc = ax.scatter(distances, bands[:, ib],
                                c=spin[:, ib], cmap=cmap, norm=norm,
                                s=8, linewidths=0, zorder=2, label=f"Exciton {ib+1}")
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$\langle S_z \rangle$")
            cbar.set_ticks([-0.5, 0, 0.5])
        else:
            colors = plt.cm.viridis(np.linspace(0, 0.85, nstates))
            for ib in range(nstates):
                ax.plot(distances, bands[:, ib], color=colors[ib],
                        lw=1.5, label=f"Exciton {ib+1}")
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


    def get_dispersion_suck(self, path):
        """ 
        Obtain dispersion along symmetry lines.
        
        Similar to band plots in k-space, check YamboExcitonDB for more comments

        :: path is instance of Path class
        """
        qpoints = self.red_qpoints
        qpath    = np.array(path.kpoints)


        rep = list(range(-1,2))
        qpoints_rep, qpoints_idx_rep = replicate_red_kmesh(qpoints,repx=rep,repy=rep,repz=[0])
        car_qpoints = red_car(qpoints_rep,lat=self.lattice.lat)

        exc_indexes = get_path(car_qpoints,self.rlat,None,path)[1] #indices are second output

        exc_qpoints  = np.array(qpoints_rep[exc_indexes])
        exc_indexes = qpoints_idx_rep[exc_indexes]
        self.exc_indexes = exc_indexes
        # Here assuming same ordering in index expansion between k-yambopy and q-yambo...
        energies = self.exc_energies[self.lattice.kpoints_indexes]
        energies_path  = energies[exc_indexes]
        
        ybs_disp = YambopyBandStructure(energies_path, exc_qpoints, kpath=path)
        return ybs_disp#, energies
        
    
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
    
    def plot_dispersion():
        """ Do plot
        """
    
    def __str__(self):
        lines = []; app = lines.append
        app(" Exciton Dispersion ")
        app(" Number of qpoints:                    %d"%self.nqpoints)
        app(" Number of exciton branches read:      %d"%self.nexcitons)
        app(" Total number of excitons/transitions: %d"%self.ntransitions)
        return "\n".join(lines)
