import numpy as np 
from yambopy.wannier.wann_H2p import *
from qepy.lattice import Path
import inspect
import matplotlib.pyplot as plt
from yambopy.lattice import calculate_distances
from yambopy.wannier.wann_occupations import TB_occupations

class ExcitonBands(H2P):
    def __init__(self, h2p: 'H2P', path_qpoints: 'Path', method="Boltz", Tel=0.0, Tbos=300.0, sigma=0.1):
        # Get the __init__ argument names of the parent class (excluding 'self')
        if not isinstance(path_qpoints, Path):
            raise TypeError('Argument must be an instance of Path')
        if not isinstance(h2p, H2P):
            raise TypeError('Argument must be an instance of H2P')
        parent_init_params = inspect.signature(H2P.__init__).parameters.keys()
        parent_init_params = [param for param in parent_init_params if param != "self"]

        # Filter attributes in the instance that match the parent's __init__ arguments
        parent_args = {key: getattr(h2p, key) for key in parent_init_params if hasattr(h2p, key)}

        # Call the parent's __init__ with the filtered arguments
        super().__init__(**parent_args)
        self.path_qpoints = path_qpoints
        self.nq_list = len(path_qpoints.get_klist())
        self.red_kpoints = self.path_qpoints.get_klist()[:,0:3]
        self.car_kpoints = red_car(self.red_kpoints, self.kmpgrid.rlat)*ang2bohr # result in Bohr
        self.nq_double = self.nq_list
        self.method = method
        self.Tel = Tel
        self.Tbos = Tbos
        self.sigma = sigma
        self.H2P = self.buildH2P_qlist()
        self.h2p = h2p

    def buildH2P_qlist(self):        
        H2P = np.zeros((self.nq_list, self.dimbse, self.dimbse), dtype=np.complex128)
        print('initialize buildh2p from cpot')
        t0 = time()
        cpot_array = None

        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            # Ensure inputs are NumPy arrays
            #K_direct = self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
            #   *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])
            cpot_array = self.cpot.v2dt2(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2dt2(np.array([[0,0,0]]),self.red_kpoints)

        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
                        # K_direct = self.cpot.v2dk(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:] )\
                        # *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])
            cpot_array = self.cpot.v2dk(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2dk(np.array([[0,0,0]]),self.red_kpoints)

        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            cpot_array = self.cpot.vcoul(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.vcoul(np.array([[0,0,0]]),self.red_kpoints)

        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            cpot_array = self.cpot.v2dt(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2dt(np.array([[0,0,0]]),self.red_kpoints)

        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            cpot_array = self.cpot.v2drk(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            cpot_q_array = self.cpot.v2drk(np.array([[0,0,0]]),self.red_kpoints)

        kminusqlist_table = self.kmpgrid.k[:,None,:] - self.red_kpoints[None,:,:]
        eigv_kmq, eigvec_kmq = self.model.get_eigenval_and_vec(kminusqlist_table.reshape(self.nk*self.nq_list,3))
        # compute the fermi occupations for k-q
        f_kmqn = self._get_occupations(eigv_kmq, self.model.fermie) #Fermi-dirac occupations in shape of eigv_kmq 
        f_kmqn = TB_occupations(eigv_kmq,Tel = self.Tel, Tbos=self.Tbos, Eb=self.model.fermie, sigma=self.sigma, fermie=self.model.fermie)._get_fkn(method=self.method)
        eigv_kmq = np.array(eigv_kmq).reshape(self.nk, self.nq_list, self.nb)
        self.f_kmqn = f_kmqn.reshape(self.nk, self.nq_list, self.nb)    # these reshapes are not very nice
        eigvec_kmq = np.array(eigvec_kmq).reshape(self.nk, self.nq_list, self.nb, self.nb)
        eigv_k = self.eigv
        eigvec_k = self.eigvec
        
        eigc = eigvec_k[self.BSE_table[:,0], :, self.BSE_table[:,2]][:,np.newaxis,:]   # conduction bands
        eigcp = eigvec_k[self.BSE_table[:,0], :, self.BSE_table[:,2]][np.newaxis,:,:]   # conduction bands prime
        eigv = eigvec_kmq[self.BSE_table[:,0],:,:,self.BSE_table[:,1]][:,np.newaxis,:,:]  # Valence bands of ikminusq
        eigvp = eigvec_kmq[self.BSE_table[:,0],:,:,self.BSE_table[:,1]][np.newaxis,:,:,:]  # Valence bands prime of ikminusq
        dotc = np.einsum('ijk,ijk->ij',np.conjugate(eigc), eigcp)
        dotv = np.einsum('ijkl,ijkl->kij',np.conjugate(eigvp), eigv)

        dotc2 = np.einsum('ijk,ijlk->li',np.conjugate(eigc), eigv)
        dotv2 = np.einsum('ijlk,ijk->lj',np.conjugate(eigvp), eigcp)
                
        del eigc, eigcp, eigv, eigvp
        gc.collect()
        
        K_direct = cpot_array[self.BSE_table[:,0],][:,self.BSE_table[:,0]] * dotc * dotv
        del dotc, dotv
        gc.collect()
        K_Ex = - cpot_q_array.T * dotc2 * dotv2
        K_sum = K_direct + K_Ex[:,:,np.newaxis]

        f_diff = (self.f_kn[self.BSE_table[:,0],:][:,self.BSE_table[:,1]][None,:,:]-self.f_kmqn[self.BSE_table[:,0],:,:][:,:,self.BSE_table[:,2]].swapaxes(1,0))
        self.K_direct = K_direct
        self.K_Ex = K_Ex
        gc.collect()

        H2P = f_diff/self.nk * K_sum

        result = eigv_kmq[self.BSE_table[:, 0],:, self.BSE_table[:, 1]].T  # Shape: (nqpoints, ntransitions)
        eigv_diff = self.eigv[self.BSE_table[:,0],self.BSE_table[:,2]] - result
        diag = np.einsum('ij,ki->kij', np.eye(self.dimbse),eigv_diff)  # when t ==tp
        H2P += diag
        print(f'Completed in {time() - t0} seconds')

        return H2P

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
    
    def plot_exciton_dispersion(self, n = None, ax=None, **kwargs):
        """Take the weights from the eigenvectors, and plot first n exciton bands"""
        if n is None: n = self.dimbse        
        if  not hasattr(self, 'h2peigvec_vck'): self.solve_H2P()
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
    
    def plot_exciton_2D_ax(self,ax,q=0,n=None,f=None,mode='hexagon',limfactor=0.8,spin_pol=None,**kwargs):
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
