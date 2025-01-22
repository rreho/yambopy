import numpy as np 
from yambopy.wannier.wann_H2p import *
from qepy.lattice import Path
import inspect
class ExcitonBands(H2P):
    def __init__(self, h2p: 'H2P', path_qpoints: 'Path'):
        super().__init__(**vars(h2p))
        if not isinstance(path_qpoints, Path):
            raise TypeError('Argument must be an instance of Path')
        if not isinstance(h2p, H2P):
            raise TypeError('Argument must be an instance of H2P')
        self.path_qpoints = path_qpoints
        self.nq_list = len(path_qpoints.get_klist())
        self.red_kpoints = self.path_qpoints.get_klist()[:,0:3]
        self.car_kpoints = red_car(self.red_kpoints, self.kmpgrid.rlat)*ang2bohr # result in Bohr

    def _buildH2P_qlist(self):        
            H2P = np.zeros((self.nq_list, self.dimbse, self.dimbse), dtype=np.complex128)
            print('initialize buildh2p from cpot')
            t0 = time()
            kminusqlist_table = self.kmpgrid.k[:,None,:] - self.red_kpoints[None,:,:]
            eigv_kmq, eigvec_kmq = self.model.get_eigenval_and_vec(kminusqlist_table.reshape(self.nk*self.nq_list,3))
            # compute the fermi occupations for k-q
            f_kmqn = self._get_occupations(self.nq_list, self.nb, self.eigv_kmq, self.model.fermie)
            eigv_kmq = np.array(eigv_kmq).reshape(self.nk, self.nq_list, self.nb)
            self.f_kmqn = f_kmqn.reshape(self.nk, self.nq_list, self.nb)
            eigvec_kmq = np.array(eigvec_kmq).reshape(self.nk, self.nq_list, self.nb, self.nb)
            eigv_k = self.eigv
            eigvec_k = self.eigvec

            eigc1 = eigvec_k[:, :, self.BSE_table[:,2]][:,:,np.newaxis, :]   # conduction bands
            eigc2 = eigvec_k[:, :, self.BSE_table[:,2]][:,:,:,np.newaxis]   # conduction bands
            eigv1 = eigvec_kmq[:, :, :, self.BSE_table[:,1]][:, :, :,  : ]  # Valence bands
            eigv2 = eigvec_kmq[:, :, :, self.BSE_table[:,1]][:, :, :, :]  # Valence bands

            dotc = np.einsum('ijkl,ijkl->ikl',np.conjugate(eigc1), eigc2)
            dotv = np.einsum('ijkl, ijkm -> jilm ',np.conjugate(eigv1), eigv2)
            dotc2 = np.einsum('ijkl,imjo->mlo',np.conjugate(eigc1), eigv2)
            dotv2 = np.einsum('ijkl,ikop->jlo',np.conjugate(eigv1), eigc2)

            v2dt2_array_ex = self.cpot.v2dt2(self.car_kpoints,np.array([[0.0,0.0,0.0]]))
            v2dt2_array_d  = self.cpot.v2dt2(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            #K_direct = self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
            #   *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])
            K_d = np.einsum('ij, ilm, njlm ->nlm' , v2dt2_array_d, dotc, dotv)
            ## Ex term
            K_ex = np.einsum('ab, cde, cde -> cde', v2dt2_array_ex, dotc2, dotv2)
            H2P = K_d - K_ex
            eigv_diff = (eigv_k[:,None, self.BSE_table[:,2]]-eigv_kmq[:,:,self.BSE_table[:,1]])[self.BSE_table[:,0],:,:].swapaxes(1,0)
            diag = np.einsum('ij,ki->kij', np.eye(self.dimbse), eigv_diff)  # when t ==tp
            H2P += diag
            print(f'Completed in {time() - t0} seconds')
            return H2P        

    def _get_occupations(self, nk, nb, eigv, fermie):
        occupations = np.zeros((nk, nb))
        occupations = fermi_dirac(eigv,fermie)
        return np.real(occupations)
