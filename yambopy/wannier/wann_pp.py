import numpy as np 
from yambopy.wannier.wann_H2p import *
from qepy.lattice import Path

class ExcitonBands(H2P):
    def __init__(self, h2p: 'H2P', path_qpoints: 'Path'):
        super().__init__(**h2p.args)
        if not isinstance(path_qpoints, Path):
            raise TypeError('Argument must be an instance of Path')
        if not isinstance(h2p, H2P):
            raise TypeError('Argument must be an instance of H2P')
        self.path_qpoints = path_qpoints
        self.nq_list = len(path_qpoints.get_klist())
        self.red_kpoints = self.path_qpoints.get_klist()[:,0:3]

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

            # Precompute kplusq and kminusq tables
            ikminusq = self.kminusq_table[:, :, 1]
            ikminusgamma = self.kminusq_table[:, :, 0]
            iqminusgamma = self.qminusk_table[:,:,0]
            iqminusk = self.qminusk_table[:,:,1]
            eigc1 = eigvec_k[self.BSE_table[:,0], :, self.BSE_table[:,2]][:,np.newaxis,:]   # conduction bands
            eigc2 = eigvec_k[self.BSE_table[:,0], :, self.BSE_table[:,2]][np.newaxis,:,:]   # conduction bands
            eigv1 = eigvec_kmq[self.BSE_table[:,0], :, :, self.BSE_table[:,1]][:, :, :, :, np.newaxis]  # Valence bands
            eigv2 = eigvec_kmq[self.BSE_table[:,0], :, :, self.BSE_table[:,1]][:, :, :, np.newaxis, :]  # Valence bands
            
            dotc = np.einsum('ijk,ijk->ij',np.conjugate(eigc1), eigc2)
            dotv = np.einsum('ijklm,ijkml->ij',np.conjugate(eigv1), eigv2)
            v2dt2_array = self._getKdq(0,0,0,0,0,0,0)       #sorry
            #K_direct = self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
            #   *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])
            K_direct = v2dt2_array[self.BSE_table[:,0],][:,self.BSE_table[:,0]] * dotc*dotv
            ## Ex term
            dotc2 = np.einsum('ijk,jilk->li',np.conjugate(eigc1), eigv2)
            dotv2 = np.einsum('ijkl,jil->ki',np.conjugate(eigv1), eigc2)
            K_Ex = v2dt2_array[0][self.BSE_table[:,0]] * dotc2 * dotv2
            K_diff = K_direct - K_Ex[:,np.newaxis,:]
            f_diff = self.f_kmqn[self.BSE_table[:,0],:, self.BSE_table[:,1]] \
                -  self.f_kn[:,:][self.BSE_table[:,0], None, self.BSE_table[:,2]] 
            H2P = f_diff * K_diff
            result = eigv_kmq[self.BSE_table[:, 0], :, self.BSE_table[:, 1]]  # Shape: (ntrasitions_k, nq, ntransitions_v)
            eigv_diff = eigv_k[self.BSE_table[:,0],None,self.BSE_table[:,2]] - result # Shape: (ntrasitions_k, nq, ntransitions_c) - (ntrasitions_k, nq, ntransitions_v)
            self.eigv_diff_ttp = eigv_diff
            self.eigvecc_t = eigc1[:,0,:]
            self.eigvecv_t = eigv1[:,0,0,:]
            diag = np.einsum('ij,ki->kij', np.eye(self.dimbse), eigv_diff)  # when t ==tp
            H2P += diag
            print(f'Completed in {time() - t0} seconds')
            return H2P        

    def _get_occupations(self, nk, nb, eigv, fermie):
        occupations = np.zeros((nk, nb))
        occupations = fermi_dirac(eigv,fermie)
        return np.real(occupations)
