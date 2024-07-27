import numpy as np
from yambopy.wannier.wann_Gfuncs import GreensFunctions
from yambopy.wannier.wann_io import RMN
from yambopy.wannier.wann_utils import *


class TB_dipoles():
    '''dipoles = 1/(\DeltaE+ieta)*<c,k|P_\alpha|v,k>'''
    def __init__(self , nc, nv, bse_nc, bse_nv, nkpoints, eigv, eigvec, \
                 eta, hlm, T_table, BSE_table, h2peigvec = None, method = 'real', with_bse=True, rmn = None):
        # hk, hlm are TBMODEL hamiltonians
        self.ntransitions = nc*nv*nkpoints
        self.nbsetransitions = bse_nc*bse_nv*nkpoints
        self.nc = nc
        self.nv = nv
        self.nkpoints = nkpoints
        self.eigv = eigv
        self.eigvec = eigvec
        self.nb = nc+nv
        self.bse_nv = bse_nv
        self.bse_nc = bse_nc
        self.bse_nb = bse_nv + bse_nc
        # self.eigvv = eigvv
        # self.eigvc = eigvc
        # self.eigvecv = eigvecv
        # self.eigvecc = eigvecc
        self.eta = eta
        self.hlm = hlm
        if self.hlm[0][0][0][0] ==0:
            raise ValueError
        else:
            print(f"hlm not zero: {self.hlm[0][0][0][0]}")
        self.method = method
        if(rmn is not None):
            self.rmn = rmn
            self.method = 'position'
        #T_table = [transition, ik, iv, ic] 
        self.T_table = T_table
        #[nkpoints,3,nbands,nbands]
        # self.dipoles = self._get_dipoles(method)
        # self.d_knm = self._get_dipoles_nm(method)
        if (h2peigvec is not None):
            self.h2peigvec = h2peigvec
            # self.dipoles_bse = self._get_dipoles_bse(method)
        if with_bse:
            self._get_dipoles_bse(method=method)
            self.T_table = BSE_table
        else:
            self._get_dipoles(method=method)
        self._get_osc_strength(method)

    # full dipoles matrix, not only cv, needs adaptation
    def _get_dipoles_nm(self, method):
        if (method == 'real'):
            dipoles = np.zeros((self.nkpoints, self.nb,self.nb,3),dtype=np.complex128)
            for n in range(0, self.nb):
                for m in range(0,self.nb):
                    for ik in range(0,self.nkpoints):
                        # E = self.eigv[ik, n]-self.eigv[ik, m]
                        # GR = GreensFunctions(E,0,self.eta).GR
                        #GA = GreensFunctions(E,0,self.eta).GA
                        # hlm is Bohr*eV
                        dipoles[ik, n, m,0] = np.vdot(self.eigvec[ik,:,n],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,m]))
                        dipoles[ik, n, m,1] = np.vdot(self.eigvec[ik,:,n],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,m]))
                        dipoles[ik, n, m,2] = np.vdot(self.eigvec[ik,:,n],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,m]))

        return dipoles#/(HA2EV**3)
    
    def _get_dipoles(self, method):
        if method == 'real':  # Parallelize over kpoints
            import time
            print("Starting dipole matrix formation.\n")
            t0 = time.time()

            # Determine the dimension of hlm
            dim_hlm = 3 if np.count_nonzero(self.hlm[:,:,:,2]) > 0 else 2

            # Extract k, v, c from T_table
            k_indices, v_indices, c_indices = self.T_table.T

            # Compute Green's function for all transitions
            w = self.eigv[k_indices, c_indices]
            E = self.eigv[k_indices, v_indices]
            GR = GreensFunctions(w=w, E=E, eta=self.eta).GR

            # Initialize dipoles array
            dipoles = np.zeros((self.ntransitions, dim_hlm), dtype=np.complex128)

            # Prepare eigenvectors
            eigvec_c = self.eigvec[k_indices, :, c_indices]
            eigvec_v = self.eigvec[k_indices, :, v_indices]

            # Compute dipoles
            for dim in range(dim_hlm):
                # Compute the dot product
                dot_product = np.einsum('ij,ijk->ik', eigvec_v, self.hlm[k_indices, :, :, dim])
                
                # Compute the vdot and multiply with GR
                dipoles[:, dim] = GR * np.sum(np.conjugate(eigvec_c) * dot_product, axis=1)

            # Reshape dipoles to match your original shape
            final_dipoles = np.zeros((self.ntransitions, self.nkpoints, self.nb, self.nb, 3), dtype=np.complex128)
            final_dipoles[np.arange(self.ntransitions), k_indices, c_indices, v_indices, :dim_hlm] = dipoles
            self.dipoles = final_dipoles / (HA2EV ** 3)
            print("Dipoles matrix computed successfully in serial mode.")
            print(f"Time for Dipoles matrix formation: {time.time() - t0:.2f}")
        if (method == 'yambo'):
            dipoles = np.zeros((self.ntransitions, self.nkpoints, self.nb,self.nb,3),dtype=np.complex128)
            for n in range(0, self.ntransitions):
                for t in self.T_table:
                    ik = t[0]
                    iv = t[1]
                    ic = t[2]
                    # here I want 1/(E_cv-E_vk) so w=\DeltaE and E = 0 in the call to GFs
                    # E = self.eigv[ik, ic]-self.eigv[ik, iv]
                    GR = GreensFunctions(w=self.eigv[ik, ic], E=self.eigv[ik, iv], eta=self.eta).GR #w - E
                    GA = GreensFunctions(w=self.eigv[ik, ic], E=self.eigv[ik, iv], eta=self.eta).GA #w - E
                    dipoles[n, ik, ic, iv, 0] = (GR+GA)*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                    dipoles[n, ik, ic, iv, 1] = (GR+GA)*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                    dipoles[n, ik, ic, iv, 2] = (GR+GA)*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))
            self.dipoles = dipoles/(HA2EV**3)  

        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
           

    def _get_dipoles_bse(self, method):
        if (method == 'real'):
            dipoles = np.zeros((self.nbsetransitions, self.nkpoints, self.bse_nb,self.bse_nb,3),dtype=np.complex128)
            
            for n in range(0,self.nbsetransitions):
                for t in self.T_table:
                    ik = t[0]
                    iv = t[1]
                    ic = t[2]
                    # here I want 1/(E_cv-E_vk) so w=\DeltaE and E = 0 in the call to GFs
                    E = self.eigv[ik, ic]-self.eigv[ik, iv]
                    GR = GreensFunctions(E,0,self.eta).GR
                    # GA = GreensFunctions(E,0,self.eta).GA
                    dipoles[n, ik, ic-self.nv, self.bse_nv-self.nv+iv, 0] = GR*self.h2peigvec[t,self.bse_nv-self.nv+iv,ic-self.nv,ik]*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                    dipoles[n, ik, ic-self.nv, self.bse_nv-self.nv+iv, 1] = GR*self.h2peigvec[t,self.bse_nv-self.nv+iv,ic-self.nv,ik]*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                    dipoles[n, ik, ic-self.nv, self.bse_nv-self.nv+iv, 2] = GR*self.h2peigvec[t,self.bse_nv-self.nv+iv,ic-self.nv,ik]*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))
        
        if (method == 'yambo'):
            dipoles = np.zeros((self.ntransitions, self.nkpoints, self.nb,self.nb,3),dtype=np.complex128)
            for n in range(0, self.ntransitions):
                for t in self.T_table:
                    ik = t[0]
                    iv = t[1]
                    ic = t[2]
                    # here I want 1/(E_cv-E_vk) so w=\DeltaE and E = 0 in the call to GFs
                    E = self.eigv[ik, ic]-self.eigv[ik, iv]
                    GR = GreensFunctions(E,0,self.eta).GR
                    GA = GreensFunctions(E,0,self.eta).GA
                    dipoles[n, ik, ic, iv,0] = (GR+GA)*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                    dipoles[n, ik, ic, iv,1] = (GR+GA)*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                    dipoles[n, ik, ic, iv,2] = (GR+GA)*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))      
        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        self.dipoles = dipoles                        
    
    def _get_osc_strength(self,method):
        '''computes osc strength from dipoles'''
        print('Computing oscillator strenght')
        import time
        t0 = time.time()
        dipoles = self.dipoles
        F_kcv = np.zeros((self.ntransitions, 3, 3), dtype=np.complex128)   
        print(f"nonzero in dipoles: {np.count_nonzero(self.dipoles[:,:,:,:,0])}")
        print(f"nonzero in dipoles: {np.count_nonzero(self.dipoles[:,:,:,:,1])}")
        print(f"nonzero in dipoles: {np.count_nonzero(self.dipoles[:,:,:,:,2])}")

        if (method == 'real'):
            for i, t in enumerate(dipoles):
                ik = self.T_table[i][0]
                iv = self.T_table[i][1]
                ic = self.T_table[i][2]
                factorRx = t[ik, ic,iv, 0]
                factorLx = factorRx.conj() 
                factorRy = t[ik, ic,iv, 1]
                factorLy = factorRy.conj() 
                factorRz = t[ik, ic-self.nv, self.bse_nv-self.nv+iv, 2]
                factorLz = factorRz.conj() 
                F_kcv[i,0,0] = factorRx*factorLx
                F_kcv[i,0,1] = factorRx*factorLy
                F_kcv[i,0,2] = factorRx*factorLz
                F_kcv[i,1,0] = factorRy*factorLx
                F_kcv[i,1,1] = factorRy*factorLy
                F_kcv[i,1,2] = factorRy*factorLz
                F_kcv[i,2,0] = factorRz*factorLx
                F_kcv[i,2,1] = factorRz*factorLy
                F_kcv[i,2,2] = factorRz*factorLz

        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        self.F_kcv = F_kcv        
        print(f"Oscillation strenght computed succesfully in {time.time()-t0:.2f}s")