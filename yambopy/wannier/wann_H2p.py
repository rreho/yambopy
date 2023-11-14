import numpy as np
from yambopy.wannier.wann_tb_mp import tb_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
class H2P():
    '''Build the 2-particle resonant Hamiltonian H2P'''
    def __init__(self, nk, nb, nc, nv,eigv, T_table, lat, kmpgrid, qmpgrid,excitons=None, kernel=None,cpot='v2dt', method='model'): 
        self.nk = nk
        self.nb = nb
        self.nc = nc
        self.nv = nv
        self.eigv = eigv
        self.kmpgrid = kmpgrid
        self.qmpgrid = qmpgrid
        self.dimbse = nk*nc*nv
        self.lat = lat
        # for now Transitions are the same as dipoles?
        self.T_table = T_table
        self.method = method
        if(self.method=='model' and cpot is not None):
            print('\n Building H2P from model Coulomb potentials. Default is v2dt\n')
            self.cpot=self.cpot
            self.H2P = self._buildH2P_fromcpot()
        elif(self.method=='kernel' and cpot is None):
            print('\n Building H2P from model YamboKernelDB\n')
            #Remember that the BSEtable in Yambopy start counting from 1 and not to 0
            self.kernel = self._buildKernel(kernel)
            self.excitons = self._getexcitons(excitons)
            self.H2P = self._buildH2P()
        else:
            print('\nWarning! Kernel can be built only from Yambo database or model Coulomb potential\n')


    def _buildH2P(self,cpot):
        H2P = np.zeros(self.dimbse,self.dimbse)
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
        H2P = np.zeros(self.dimbse,self.dimbse)
        for t in range(self.dimbse):
            for tp in range(self.dimbse):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                ikp = self.T_table[tp][0]
                ivp = self.T_table[tp][1]
                icp = self.T_table[tp][2]
                if (t == tp):
                    H2P[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv] + self.cpot(self.lat.car_kpoints[ikp,:]-self.lat.car_kpoints[ik,:])
                else:
                    H2P[t,tp] = self.cpot(self.lat.car_kpoints[ikp,:]-self.lat.car_kpoints[ik,:])
        return H2P
    
    def _buildKernel(self, kernel):
        try:
            kernel = kernel
        except  TypeError:
            print('Error Kernel is None')
        return kernel
        
    def _getexcitons(self, excitons):
        try:
            kernel = kernel
        except  TypeError:
            print('Error YamboExcitonDB is None')
        return excitons
        
                