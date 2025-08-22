import numpy as np 
from yambopy.wannier.wann_H2p import *
from qepy.lattice import Path
import inspect
import matplotlib.pyplot as plt
from yambopy.lattice import calculate_distances
from yambopy.wannier.wann_occupations import TB_occupations

class ExcitonBands(H2P):
    def __init__(self, h2p: 'H2P', path_qpoints: 'Path', method="Boltz", Tel=0.0, Tbos=300.0, sigma=0.1, memmap_path=None, distributed=False, keep_kernels=False):
        # Get the __init__ argument names of the parent class (excluding 'self')
        self.kmpgrid = h2p.kmpgrid
        self.cpot = h2p.cpot
        self.ctype = h2p.ctype
        self.dimbse = h2p.dimbse
        self.model = h2p.model
        self.nk = h2p.nk
        self.nb = h2p.nb
        self.eigv = h2p.eigv
        self.eigvec = h2p.eigvec
        self.BSE_table = h2p.BSE_table
        self.f_kn = h2p.f_kn
        self.bse_nc = h2p.bse_nc
        self.bse_nv = h2p.bse_nv
        self.nv = h2p.nv
        self.nc = h2p.nc
        self.ntransitions = h2p.ntransitions
        self.latdb = h2p.latdb
        self.qmpgrid = h2p.qmpgrid        
        self.method = method
        self.Tel = Tel
        self.Tbos = Tbos
        self.sigma = sigma
        self.H2P = h2p.H2P
        self.h2p = h2p
        if not isinstance(path_qpoints, Path):
            print('No Path was provided, gamma point only')
        else:
            
            self.path_qpoints = path_qpoints
            self.nq_list = len(path_qpoints.get_klist())
            self.nq_double = self.nq_list
            self.red_kpoints = self.path_qpoints.get_klist()[:,0:3]
            self.car_kpoints = red_car(self.red_kpoints, self.kmpgrid.rlat)#*ang2bohr # result in Bohr
            self.H2P = self.buildH2P_qlist(memmap_path=memmap_path, distributed=distributed, keep_kernels=keep_kernels)
        if not isinstance(h2p, H2P):
            raise TypeError('Argument must be an instance of H2P')
        
    def buildH2P_qlist(self, memmap_path=None, distributed=False, keep_kernels=False):
        """
        Memory-safe, optionally multi-node build of H2P over q-list.

        Parameters
        ----------
        memmap_path : str or None
            If given, H2P is created as a memmap at this path. Strongly recommended for large problems.
        distributed : bool
            If True and mpi4py is available, split q's across MPI ranks and write into the same memmap.
            Each rank writes disjoint q-slices; rank 0 returns the array handle, others return None.
        keep_kernels : bool
            If True, stores self.K_direct and self.K_Ex for diagnostics (uses extra RAM).

        Returns
        -------
        H2P : np.ndarray or np.memmap on rank 0; None on other ranks if distributed=True.
            Shape (nq_list, dimbse, dimbse), dtype=complex128.
        """
        import numpy as np, gc
        from time import time

        t0 = time()

        # --- MPI setup (optional) ---
        comm = None
        rank = 0
        size = 1
        if distributed:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
            except Exception:
                # fallback to single-process if mpi4py not available
                comm = None
                rank = 0
                size = 1

        nq   = int(self.nq_list)
        dim  = int(self.dimbse)
        nb   = int(self.nb)
        nk   = int(self.nk)

        # ---- Coulomb matrices (small enough; build once) ----
        if   self.ctype == 'v2dt2':
            cpot_array   = self.cpot.v2dt2(self.kmpgrid.car_kpoints, self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2dt2(np.array([[0,0,0]]),      self.red_kpoints).ravel()
        elif self.ctype == 'v2dk':
            cpot_array   = self.cpot.v2dk(self.kmpgrid.car_kpoints, self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2dk(np.array([[0,0,0]]),      self.red_kpoints).ravel()
        elif self.ctype == 'vcoul':
            cpot_array   = self.cpot.vcoul(self.kmpgrid.car_kpoints, self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.vcoul(np.array([[0,0,0]]),      self.red_kpoints).ravel()
        elif self.ctype == 'v2dt':
            cpot_array   = self.cpot.v2dt(self.kmpgrid.car_kpoints, self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2dt(np.array([[0,0,0]]),      self.red_kpoints).ravel()
        elif self.ctype == 'v2drk':
            cpot_array   = self.cpot.v2drk(self.kmpgrid.car_kpoints, self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2drk(np.array([[0,0,0]]),      self.red_kpoints).ravel()
        else:
            raise ValueError(f"Unknown ctype={self.ctype}")

        # Map k-indices used by transitions
        ik  = self.BSE_table[:, 0].astype(np.intp)          # (dim,)
        ivb = self.BSE_table[:, 1].astype(np.intp)          # (dim,)
        icb = self.BSE_table[:, 2].astype(np.intp)          # (dim,)

        # k − q list
        kminusqlist = self.kmpgrid.k[:, None, :] - self.red_kpoints[None, :, :]   # (nk, nq, 3)

        # --- Precompute k-only stuff once (tiny) ---
        # Conduction eigenvectors at k for each transition t = (ik, icb)
        C_k = self.eigvec[ik, :, icb]            # (dim, nb)
        # Gram over bands for conduction part (constant over q)
        # dotc[i,j] = <C_i | C_j> = sum_b conj(C_i[b]) * C_j[b]
        dotc = C_k.conj() @ C_k.T                # (dim, dim)

        # Coulomb map for transitions (dim x dim)
        cpot_tt = cpot_array[ik][:, ik]          # (dim, dim)

        # Valence occupations at k for each transition (constant over q)
        f_val = self.f_kn[ik, ivb]               # (dim,)

        # --- Output array: memmap or in-memory ---
        if memmap_path is not None:
            # Create/overwrite memmap on every rank; disjoint writes are safe
            H2P_out = np.memmap(memmap_path, dtype=np.complex128, mode='w+', shape=(nq, dim, dim))
        else:
            # Only root holds full array if not memmapping
            H2P_out = np.zeros((nq, dim, dim), dtype=np.complex128) if rank == 0 else None

        # --- Distribute q indices across ranks ---
        q_indices = np.arange(nq, dtype=np.intp)[rank::size]

        # --- Main loop over q (streaming; O(1) memory per q) ---
        for iq in q_indices:
            # 1) Eigenpairs at k - q_i (only this q)
            k_minus_q = kminusqlist[:, iq, :]                              # (nk, 3)
            eigv_kmq_iq, eigvec_kmq_iq = self.model.get_eigenval_and_vec(k_minus_q)  # eigv: (nk, nb), eigvec: (nk, nb, nb)

            # 2) Fermi occupations at k - q_i (only this q)
            f_kmq_iq = TB_occupations(
                eigv_kmq_iq, Tel=self.Tel, Tbos=self.Tbos, Eb=self.model.fermie,
                sigma=self.sigma, fermie=self.model.fermie
            )._get_fkn(method=self.method)                                 # (nk, nb)

            # 3) Build valence vectors V_i(q): for each transition t=(ik, ivb)
            V_q = eigvec_kmq_iq[ik, :, ivb]                                # (dim, nb)

            # 4) dotv(q): Gram over valence vectors -> (dim, dim)
            #     dotv[i,j] = <V_i(q)|V_j(q)>
            dotv = V_q.conj() @ V_q.T                                      # (dim, dim)

            # 5) Exchange side vectors (length dim), per transition
            #     dotc2[i] = <C_i | V_i(q)>,   dotv2[j] = <V_j(q) | C_j>
            dotc2 = np.einsum('ib,ib->i', C_k.conj(), V_q, optimize=True)  # (dim,)
            dotv2 = np.einsum('jb,jb->j', V_q.conj(), C_k, optimize=True)  # (dim,)

            # 6) Direct + exchange kernels at this q
            #     K_direct(q) = cpot_tt * dotc * dotv(q)        (elementwise)
            K_direct_q = cpot_tt * dotc * dotv                                 # (dim, dim)

            #     K_ex(q) = - v(q) * [dotc2 ⊗ dotv2]        (outer product)
            #     If you need legacy broadcasting semantics, replace outer with replication.
            K_ex_q = - cpot_q_array[iq] * (dotc2[:, None] * dotv2[None, :])    # (dim, dim)

            if keep_kernels and rank == 0:
                # Stash small views for debugging (optional)
                # WARNING: keeping full K_* over all q re-introduces memory growth.
                if iq == 0:
                    self.K_direct = np.empty((nq, dim, dim), dtype=np.complex128)
                    self.K_Ex     = np.empty((nq, dim, dim), dtype=np.complex128)
                self.K_direct[iq] = K_direct_q
                self.K_Ex[iq]     = K_ex_q

            # 7) Occupation difference per (i,j) at this q
            #     f_diff_ij(q) = f_val[i] - f_con(q)[j]
            f_con_q = f_kmq_iq[ik, icb]                                     # (dim,)
            f_diff_q = f_val[:, None] - f_con_q[None, :]                    # (dim, dim)

            # 8) Assemble H2P(q)
            H_q = f_diff_q  * (K_direct_q + K_ex_q)              # (dim, dim)

            # 9) Add diagonal term: ΔE_i(q) on the diagonal
            #     ΔE_i(q) = E_c(ik,icb) - E_v(ik−q,ivb)
            Ec_i = self.eigv[ik, icb]                                       # (dim,)
            Ev_iq = eigv_kmq_iq[ik, ivb]                                    # (dim,)
            dE_iq = Ec_i - Ev_iq                                            # (dim,)
            H_q[np.diag_indices(dim)] += dE_iq

            # 10) Store H2P(q) into output (memmap or in-memory)
            if memmap_path is not None:
                H2P_out[iq, :, :] = H_q
            elif rank == 0:
                H2P_out[iq, :, :] = H_q

            # 11) Free per-q temporaries
            del eigv_kmq_iq, eigvec_kmq_iq, f_kmq_iq, V_q, dotv, dotc2, dotv2, K_direct_q, K_ex_q, f_con_q, f_diff_q, H_q
            gc.collect()

        # --- Synchronize and return ---
        if comm is not None:
            comm.Barrier()
        if memmap_path is not None:
            H2P_out.flush()

        if rank == 0:
            print(f'Completed in {time() - t0:.3f} s (streaming over q, distributed={distributed}, memmap={"yes" if memmap_path else "no"})')
            return H2P_out
        else:
            # non-root returns nothing in distributed mode
            return None


    def _get_occupations(self, eigv, fermie):
        occupations = fermi_dirac(eigv,fermie)
        return np.real(occupations)
    
    def get_exciton_weights_bz(self, n=None):
        """get weight of state in each band in full bz"""
        nq = self.h2p.nq
        weight = np.abs(self.h2peigvec_vck)
        weight_norm = weight/np.max(weight)

        return weight_norm[:,:n,:n] if n is not None else weight_norm
    
    def get_exciton_weights(self,nq):
        """get weight of state in each band in the q path"""
        weight = np.abs(self.h2peigvec_vck.reshape(nq,self.ntransitions, self.ntransitions))
        weight_norm = weight/np.max(weight)
        return weight_norm
    
    def solve_exc(self, n_threads=None, n_exc=None, driver='evd'):
        import contextlib
        from threadpoolctl import threadpool_limits
        ctx = threadpool_limits(limits=n_threads, user_api='blas') if n_threads else contextlib.nullcontext()
        with ctx:
            if n_exc is None: n_exc = self.dimbse        
            if  not hasattr(self, 'h2peigvec_vck'): self.solve_H2P(n_threads=n_threads, k=n_exc, driver=driver)

    def plot_exciton_dispersion(self,n=None, ax=None, **kwargs):
        """Take the weights from the eigenvectors, and plot first n exciton bands"""
        if not ax:
            fig, ax = plt.subplots()
        weight_norm = self.get_exciton_weights(nq=self.nq_list)
        xpath = calculate_distances(self.path_qpoints.get_klist())
        colormap = plt.cm.viridis
        
        kwargs.setdefault('color','black')
        kwargs.setdefault('lw',2)
        kwargs.setdefault('ls','solid')
        
        for ie in range(n):
        #for j in range(len(self.h2peigv) -1):
            alpha_value = weight_norm[:,ie]
            ax.plot(
                xpath[:],
                self.h2peigv[:, ie].real,
                **kwargs,
                # c=colormap(weight_norm[j,ie]),
                # alpha=alpha_value
            )
        self.path_qpoints.set_xticks(ax)
        ax.set_ylabel(f"Energy [eV]")
        ax.set_xlim(np.min(xpath), np.max(xpath))
        ax.set_xlim(min(xpath), max(xpath))
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.vlines(self.path_qpoints.distances, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='grey', linestyles='dashed')
        # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        # cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        # cbar.set_label('Exciton weight')
        return ax

    def plot_debug_excbands(self,h2p,ax=None,tolerance=1e-2
, **kwargs):
        matching_indices_path = []
        matching_indices_h2p = []

        for i, kpoint in enumerate(self.path_qpoints.get_klist()[:,:3]):
            for j, h2p_kpoint in enumerate(self.qmpgrid.red_kpoints):
                if np.allclose(kpoint, h2p_kpoint, atol=tolerance):
                    matching_indices_path.append(i)
                    matching_indices_h2p.append(j)
        
        eigv_matching = h2p.h2peigv[matching_indices_h2p]
        xpath = calculate_distances(self.path_qpoints.get_klist())
        excdiff = (self.h2peigv[np.array(matching_indices_path)] - eigv_matching).T
        matching_x = np.array(xpath)[matching_indices_path]
        fig, ax =plt.subplots()
        for band in excdiff:
            ax.scatter(matching_indices_path, band, color='grey')

        ticks = np.array(self.path_qpoints.get_indexes()).T
        ax.set_xticks([int(tick) for tick in ticks[0]])
        ax.set_xticklabels(ticks[1])
        ax.set_title(f'Difference of eigv between h2p and excbands,'+r"$K_{{{toll}}}$ ="+f"{tolerance}")
        print("Total accumulative dif:  " ,np.sum(np.abs(excdiff)))
        fig.show()

        return fig, ax

    def get_exciton_2D(self,f=None,n=None):
        """get data of the exciton in 2D"""
        weights = self.get_exciton_weights_bz(n=n)
        #sum all the transitions
        weights_bz_sum = np.sum(weights,axis=(1,2,3))
        if f: weights_bz_sum = f(weights_bz_sum)
        from yambopy.lattice import replicate_red_kmesh
        kmesh_full, kmesh_idx = replicate_red_kmesh(self.h2p.kmpgrid.k,repx=range(-1,2),repy=range(-1,2))
        x,y = red_car(kmesh_full,self.kmpgrid.rlat)[:,:2].T
        weights_bz_sum = weights_bz_sum[:,kmesh_idx].swapaxes(0,1)
        return x,y,weights_bz_sum
    
    def plot_exciton_2D_ax(self,ax=None,q=0,n=None,f=None,mode='hexagon',limfactor=0.8,spin_pol=None,**kwargs):
        """
        Plot the exciton weights in a 2D Brillouin zone
       
           Arguments:
            excitons -> list of exciton indexes to plot
            f -> function to apply to the exciton weights. Ex. f=log will compute the 
                 log of th weight to enhance the small contributions
            mode -> possible values are 'hexagon'/'square' to use hexagons/squares as markers for the 
                    weights plot and 'rbf' to interpolate the weights using radial basis functions.
            limfactor -> factor of the lattice parameter to choose the limits of the plot 
            scale -> size of the markers
        """
        if spin_pol is not None: print('Plotting exciton mad in 2D axis for spin polarization: %s' % spin_pol)

        if spin_pol is not None:
            x,y,weights_bz_sum_up,weights_bz_sum_dw = self.get_exciton_2D_spin_pol(excitons,f=f)
        else:
            x,y,weights_bz_sum = self.get_exciton_2D(f=f,n=n)

        weights_bz_sum=weights_bz_sum[:,q]/np.max(weights_bz_sum[:,q])
        
        #filter points outside of area
        lim = np.max(self.latdb.rlat)*limfactor
        dlim = lim*1.1
        if spin_pol is not None:
            filtered_weights_up = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum_up) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
            filtered_weights_dw = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum_dw) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
            x,y,weights_bz_sum_up = np.array(filtered_weights_up).T
            x,y,weights_bz_sum_dw = np.array(filtered_weights_dw).T
        else:
            x = np.asarray(x)
            y = np.asarray(y)
            weights_bz_sum = np.asarray(weights_bz_sum)
            mask = (x > -dlim) & (x < dlim) & (y > -dlim) & (y < dlim)
            x_filtered = x[mask]
            y_filtered = y[mask]
            weights_filtered = weights_bz_sum[mask]  # Filter rows only
            # weights_bz_sum = weights_filtered
            # x,y,weights_bz_sum = np.array(filtered_weights).T
        # Add contours of BZ
        from yambopy.plot.plotting import BZ_Wigner_Seitz
        ax.add_patch(BZ_Wigner_Seitz(self.latdb))
        weights_bz_sum=weights_bz_sum[:,q]/np.max(weights_bz_sum[:,q])
        if ax is None:
            fig,ax = plt.subplots(figsize=(12,8),dpi=300)
        #filter points outside of area
        lim = np.max(self.latdb.rlat)*limfactor
        dlim = lim*1.1
        if spin_pol is not None:
            filtered_weights_up = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum_up) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
            filtered_weights_dw = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum_dw) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
            x,y,weights_bz_sum_up = np.array(filtered_weights_up).T
            x,y,weights_bz_sum_dw = np.array(filtered_weights_dw).T
        else:
            x = np.asarray(x)
            y = np.asarray(y)
            weights_bz_sum = np.asarray(weights_bz_sum)
            mask = (x > -dlim) & (x < dlim) & (y > -dlim) & (y < dlim)
            x_filtered = x[mask]
            y_filtered = y[mask]
            weights_filtered = weights_bz_sum[mask]  # Filter rows only
            # weights_bz_sum = weights_filtered
            # x,y,weights_bz_sum = np.array(filtered_weights).T
        # Add contours of BZ
        from yambopy.plot.plotting import BZ_Wigner_Seitz
        ax.add_patch(BZ_Wigner_Seitz(self.latdb))
        
        #plotting
        if mode == 'hexagon': 
            scale = kwargs.pop('scale',1)
            if spin_pol=='up':
               ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum_up,rasterized=True,**kwargs)
            elif spin_pol=='dw':
               ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum_dw,rasterized=True,**kwargs)
            else:
               ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
        elif mode == 'square': 
            scale = kwargs.pop('scale',1)
            if spin_pol=='up':
               ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum_up,rasterized=True,**kwargs)
            elif spin_pol=='dw':
               ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum_dw,rasterized=True,**kwargs)
            else:
               ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
        elif mode == 'rbf':
            from scipy.interpolate import Rbf
            npts = kwargs.pop('npts',100)
            interp_method = kwargs.pop('interp_method','bicubic')
            if spin_pol=='up':
               rbfi = Rbf(x,y,weights_bz_sum_up,function='linear')
               x = y = np.linspace(-lim,lim,npts)
               weights_bz_sum_up = np.zeros([npts,npts])
            elif spin_pol=='dw':
               rbfi = Rbf(x,y,weights_bz_sum_dw,function='linear')
               x = y = np.linspace(-lim,lim,npts)
               weights_bz_sum_dw = np.zeros([npts,npts])
            else:
               rbfi = Rbf(x,y,weights_bz_sum,function='linear')
               x = y = np.linspace(-lim,lim,npts)
               weights_bz_sum = np.zeros([npts,npts])

        for col in range(npts):
            if spin_pol=='up':
                weights_bz_sum_up[:,col] = rbfi(x,np.ones_like(x)*y[col])
            elif spin_pol=='dw':
                weights_bz_sum_dw[:,col] = rbfi(x,np.ones_like(x)*y[col])
            else:
                weights_bz_sum[:,col] = rbfi(x,np.ones_like(x)*y[col])
        # NB we have to take the transpose of the imshow data to get the correct plot
        if spin_pol=='up':
            ax.imshow(weights_bz_sum_up.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim])
        elif spin_pol=='dw':
            ax.imshow(weights_bz_sum_dw.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim])
        else:
            ax.imshow(weights_bz_sum.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim])
        title = kwargs.pop('title',str('excitons'))

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        return ax
