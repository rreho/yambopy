import numpy as np
from yambopy.wannier.wann_tb_mp import tb_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
from yambopy.wannier.wann_utils import HA2EV, sort_eig
from yambopy.wannier.wann_dipoles import TB_dipoles
from time import time
class H2P():
    '''Build the 2-particle resonant Hamiltonian H2P'''
    def __init__(self, nk, nb, nc, nv,eigv, eigvec, T_table, latdb, kmpgrid, qmpgrid,excitons=None, \
                  kernel=None,cpot=None,ctype='v2dt2',ktype='direct', method='model'): 
        self.nk = nk
        self.nb = nb
        self.nc = nc
        self.nv = nv
        self.eigv = eigv
        self.eigvec = eigvec
        self.kmpgrid = kmpgrid
        self.qmpgrid = qmpgrid
        self.dimbse = nk*nc*nv
        self.latdb = latdb
        # for now Transitions are the same as dipoles?
        self.T_table = T_table
        self.ctype = ctype
        self.ktype = ktype
        self.method = method
        if(self.method=='model' and cpot is not None):
            print('\n Building H2P from model Coulomb potentials. Default is v2dt2\n')
            self.cpot = cpot
            if (ktype == 'IP'):
                self.H2P = self._buildH2P0_fromcpot()
            else:
                self.H2P = self._buildH2P_fromcpot()
        elif(self.method=='kernel' and cpot is None):
            print('\n Building H2P from model YamboKernelDB\n')
            #Remember that the BSEtable in Yambopy start counting from 1 and not to 0
            try:
                self.kernel = kernel
            except  TypeError:
                print('Error Kernel is None')
            self.excitons = excitons
            self.H2P = self._buildH2P()
        else:
            print('\nWarning! Kernel can be built only from Yambo database or model Coulomb potential\n')


    def _buildH2P(self):
        # to-do fix inconsistencies with table
        # inconsistencies are in the k-points in mpgrid and lat.red_kpoints in Yambo
        H2P = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
        for t in range(self.dimbse):
            for tp in range(self.dimbse):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                ikp = self.T_table[tp][0]
                ivp = self.T_table[tp][1]
                icp = self.T_table[tp][2]
                if (t == tp):
                    #plus one needed in kernel for fortran counting
                    # W_cvkc'v'k'
                    H2P[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv] + self.kernel.get_kernel_value_bands(self.excitons,[iv+1,ic+1])[ik,ikp]*HA2EV
                else:
                    H2P[t,tp] = self.kernel.get_kernel_value_bands(self.excitons,[iv+1,ic+1])[ik,ikp]*HA2EV
        return H2P
    
    def _buildH2P_fromcpot(self):
        H2P = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
        for t in range(self.dimbse):
            for tp in range(self.dimbse):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                ikp = self.T_table[tp][0]
                ivp = self.T_table[tp][1]
                icp = self.T_table[tp][2]
                K_direct = self._getKd(ik,iv,ic,ikp,ivp,icp)
                if (t==tp):
                    H2P[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv] + K_direct
                else:
                    H2P[t,tp] = K_direct
        return H2P

    def _buildH2P0_fromcpot(self):
        H2P0 = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
        for t in range(self.dimbse):
            for tp in range(self.dimbse):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                ikp = self.T_table[tp][0]
                ivp = self.T_table[tp][1]
                icp = self.T_table[tp][2]
                if (t==tp):
                    H2P0[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv]
        return H2P0    
    
    def _buildKernel(self, kernel):
        pass
        
    def _getexcitons(self, excitons):
        pass
        
    def solve_H2P(self):
        print(f'\nDiagonalizing the H2P matrix with dimensions: {self.dimbse}\n')
        t0 = time()
        self.h2peigv = np.zeros((self.dimbse), dtype=np.complex128)
        self.h2peigvec = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
        h2peigv_vck = np.zeros((self.nv, self.nc, self.nk), dtype=np.complex128)
        h2peigvec_vck = np.zeros((self.dimbse,self.nv,self.nc,self.nk),dtype=np.complex128)
        (self.h2peigv, self.h2peigvec) = np.linalg.eigh(self.H2P)
        #(self.h2peigv,self.h2peigvec) = sort_eig(self.h2peigv,self.h2peigvec)
        for t in range(self.dimbse):
            ik, iv, ic = self.T_table[t]
            h2peigvec_vck[:,iv, ic-self.nv, ik] = self.H2P[:, t]   
            h2peigv_vck[iv, ic-self.nv, ik] = self.h2peigv[t]
        
        self.h2peigv_vck = h2peigv_vck        
        self.h2peigvec_vck = h2peigvec_vck
        t1 = time()

        print(f'\n Diagonalization of H2P in {t1-t0:.3f} s')
    
    def _getKd(self,ik,iv,ic,ikp,ivp,icp):
        K_direct = self.cpot.v2dt2(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:], lc = 8.0*ANG2BOHR )\
                    *np.vdot(self.eigvec[ik,ic],self.eigvec[ikp,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])
        return K_direct

    def get_eps(self, hlm, emin, emax, estep, eta):
        '''
        Compute microscopic dielectric function 
        dipole_left/right = l/r_residuals.
        \eps_{\alpha\beta} = 1 + \sum_{kcv} dipole_left*dipole_right*(GR + GA)
        '''        
        w = np.arange(emin,emax,estep,dtype=np.float32)
        F_kcv = np.zeros((self.dimbse,3,3), dtype=np.complex128)
        eps = np.zeros((len(w),3,3), dtype=np.complex128)
        for i in range(eps.shape[0]):
            np.fill_diagonal(eps[i,:,:], 1)
        # First I have to compute the dipoles, then chi = 1 + FF*lorentzian
        #dipoles = TB_dipoles(self.nk*self.nv*self.nc, self.nc, self.nv, self.nk,self.eigv,self.eigvec, eta, hlm, self.T_table).dipoles
        dipoles = TB_dipoles(self.nk*self.nv*self.nc, self.nc, self.nv, self.nk,self.eigv,self.eigvec, eta, hlm, self.T_table,h2peigvec=self.h2peigvec_vck).dipoles_bse
        for t in range(0,self.dimbse):
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
        for ies, es in enumerate(w):
            for t in range(0,self.dimbse):
                eps[ies,:,:] += 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(self.h2peigv[t]-es)/(np.abs(es-self.h2peigv[t])**2+eta**2) \
                    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-self.h2peigv[t])**2+eta**2) 
        print('Excitonic Direct Ground state: ', self.h2peigv[0], ' [eV]')
        # self.w = w
        # self.eps_0 = eps_0
        return w, eps
