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
        self.H2P = self.buildH2P_qlist()
        self.method = method
        self.Tel = Tel
        self.Tbos = Tbos
        self.sigma = sigma

    def buildH2P_qlist(self):        
        H2P = np.zeros((self.nq_list, self.dimbse, self.dimbse), dtype=np.complex128)
        print('initialize buildh2p from cpot')
        t0 = time()
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
        dotv = np.einsum('ijkl,ijkl->kij',np.conjugate(eigv), eigvp)

        dotc2 = np.einsum('ijk,jilk->li',np.conjugate(eigc), eigvp)
        dotv2 = np.einsum('ijkl,jil->ki',np.conjugate(eigv), eigcp)
                
        del eigc, eigcp, eigv, eigvp
        gc.collect()
        cpot_array = None
        if self.ctype == 'v2dk':
            cpot_array = self.cpot.v2dk(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
        elif self.ctype == 'v2dt2':
            cpot_array = self.cpot.v2dt2(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
        
        K_direct = cpot_array[self.BSE_table[:,0],][:,self.BSE_table[:,0]] * dotc * dotv
        del dotc, dotv
        gc.collect()
        K_Ex = cpot_array[0][self.BSE_table[:,0]] * dotc2 * dotv2   # is the exchange always from gamma to q?

        K_diff = K_direct - K_Ex[:,np.newaxis,:]
        
        #f (Ev,k) − f (E∆c,k+Q) with fermi-dirac distribution - method of choice.
        f_diff = (self.f_kn[self.BSE_table[:,0],:][:,self.BSE_table[:,1]][None,:,:]-self.f_kmqn[self.BSE_table[:,0],:,:][:,:,self.BSE_table[:,2]].swapaxes(1,0))    #swap axes is just reshape to get shape q, nt, nt
        del K_Ex
        gc.collect()

        # eigv_diff_ttp = eigv_diff
        H2P = f_diff * K_diff    # For the correct occupations, not just at gamma we have to take neighbouring occupations etc.
        
        eigv_diff = (eigv_k[:,None, self.BSE_table[:,2]]-eigv_kmq[:,:,self.BSE_table[:,1]])[self.BSE_table[:,0],:,:].swapaxes(1,0)
        diag = np.einsum('lm, klm -> klm', np.eye(self.dimbse), eigv_diff) # when t ==tp
        H2P += diag
        print(f'Completed in {time() - t0} seconds')

        return H2P

    def _get_occupations(self, eigv, fermie):
        occupations = fermi_dirac(eigv,fermie)
        return np.real(occupations)
    
    def get_exciton_weights(self):
        """get weight of state in each band"""
        weight = np.abs(self.h2peigvec_vck.reshape(self.nq_list,self.ntransitions, self.ntransitions))
        weight_norm = weight/np.max(weight)
        return weight_norm
    
    def plot_exciton_dispersion(self, n = None, ax=None, **kwargs):
        """Take the weights from the eigenvectors, and plot first n exciton bands"""
        if n is None: n = self.dimbse        
        if  not hasattr(self, 'h2peigvec_vck'): self.solve_H2P()
        if not ax:
            fig, ax = plt.subplots()
        weight_norm = self.get_exciton_weights()
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
        return fig, ax

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
