import numpy as np
from yambopy.wannier.wann_Gfuncs import GreensFunctions


class TB_dipoles():
    '''dipoles = 1/(\DeltaE+ieta)*<c,k|P_\alpha|v,k>'''
    def __init__(self , ntransitions, nc, nv, nkpoints, eigv, eigvec, \
                 eta, hlm, T_table, h2peigvec = None, method = 'real'):
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
        #T_table = [transition, ik, iv, ic] 
        self.T_table = T_table
        #[nkpoints,3,nbands,nbands]
        self.dipoles = self._get_dipoles(method)
        if (h2peigvec is not None):
            self.h2peigvec = h2peigvec
            self.dipoles_bse = self._get_dipoles_bse(method)

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