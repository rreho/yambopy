import numpy as np
from yambopy.wannier.wann_Gfuncs import GreensFunctions
from yambopy.wannier.wann_io import RMN

class TB_dipoles():
    '''dipoles = 1/(\DeltaE+ieta)*<c,k|P_\alpha|v,k>'''
    def __init__(self , ntransitions, nc, nv, nkpoints, eigv, eigvec, \
                 eta, hlm, T_table, h2peigvec = None, method = 'real', rmn = None):
        # hk, hlm are TBMODEL hamiltonians
        self.ntransitions = ntransitions
        self.nc = nc
        self.nv = nv
        self.nkpoints = nkpoints
        self.eigv = eigv
        self.eigvec = eigvec
        self.nb = nc+nv
        # self.eigvv = eigvv
        # self.eigvc = eigvc
        # self.eigvecv = eigvecv
        # self.eigvecc = eigvecc
        self.eta = eta
        self.hlm = hlm
        self.method = method
        if(rmn is not None):
            self.rmn = rmn
            self.method = 'position'
        #T_table = [transition, ik, iv, ic] 
        self.T_table = T_table
        #[nkpoints,3,nbands,nbands]
        self.dipoles = self._get_dipoles(method)
        if (h2peigvec is not None):
            self.h2peigvec = h2peigvec
            self.dipoles_bse = self._get_dipoles_bse(method)
            self.F_kcv = self._get_osc_strength(method)

    def _get_dipoles(self, method):
        if (method == 'real'):
            dipoles = np.zeros((self.nkpoints, self.nb,self.nb,3),dtype=np.complex128)
            for t in range(0,self.ntransitions):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                # here I want 1/(E_cv-E_vk) so w=\DeltaE and E = 0 in the call to GFs
                E = self.eigv[ik, ic]-self.eigv[ik, iv]
                GR = GreensFunctions(E,0,self.eta).GR
                #GA = GreensFunctions(E,0,self.eta).GA
                dipoles[ik, ic, iv,0] = GR*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                dipoles[ik, ic, iv,1] = GR*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                dipoles[ik, ic, iv,2] = GR*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))

        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        return dipoles     

    def _get_dipoles_bse(self, method):
        if (method == 'real'):
            dipoles = np.zeros((self.nkpoints, self.nb,self.nb,3),dtype=np.complex128)
            for t in range(0,self.ntransitions):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                # here I want 1/(E_cv-E_vk) so w=\DeltaE and E = 0 in the call to GFs
                E = self.eigv[ik, ic]-self.eigv[ik, iv]
                GR = GreensFunctions(E,0,self.eta).GR
                #GA = GreensFunctions(E,0,self.eta).GA
                dipoles[ik, ic, iv,0] = GR*self.h2peigvec[t,iv,ic-self.nv,ik]*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                dipoles[ik, ic, iv,1] = GR*self.h2peigvec[t,iv,ic-self.nv,ik]*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                dipoles[ik, ic, iv,2] = GR*self.h2peigvec[t,iv,ic-self.nv,ik]*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))
        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        return dipoles                        
    
    def _get_osc_strength(self,method):
        '''computes osc strength from dipoles'''
        F_kcv = np.zeros((self.ntransitions,3,3), dtype=np.complex128)    
        dipoles = self.dipoles_bse
        if (method == 'real'):
            for t in range(0,self.ntransitions):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                factorRx = dipoles[ik,ic,iv,0]
                factorLx = factorRx.conj() 
                factorRy = dipoles[ik,ic,iv,1]
                factorLy = factorRy.conj() 
                factorRz = dipoles[ik,ic,iv,2]
                factorLz = factorRz.conj() 
                F_kcv[t,0,0] = F_kcv[t,0,0] + factorRx*factorLx
                F_kcv[t,0,1] = F_kcv[t,0,1] + factorRx*factorLy
                F_kcv[t,0,2] = F_kcv[t,0,2] + factorRx*factorLz
                F_kcv[t,1,0] = F_kcv[t,1,0] + factorRy*factorLx
                F_kcv[t,1,1] = F_kcv[t,1,1] + factorRy*factorLy
                F_kcv[t,1,2] = F_kcv[t,1,2] + factorRy*factorLz                    
                F_kcv[t,2,0] = F_kcv[t,2,0] + factorRz*factorLx
                F_kcv[t,2,1] = F_kcv[t,2,1] + factorRz*factorLy
                F_kcv[t,2,2] = F_kcv[t,2,2] + factorRz*factorLz
        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        return F_kcv        