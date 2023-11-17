import numpy as np
from yambopy.wannier.wann_tb_mp import tb_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
from yambopy.wannier.wann_utils import HA2EV, sort_eig
from yambopy.wannier.wann_dipoles import TB_dipoles
from yambopy.wannier.wann_occupations import TB_occupations
from time import time

class H2P():
    '''Build the 2-particle resonant Hamiltonian H2P'''
    def __init__(self, nk, nb, nc, nv,eigv, eigvec, T_table, latdb, kmpgrid, qmpgrid,excitons=None, \
                  kernel=None,cpot=None,ctype='v2dt2',ktype='direct', method='model',f_kn=None, \
                  TD=False,  TBos=300): 
        self.nk = nk
        self.nb = nb
        self.nc = nc
        self.nv = nv
        self.nq = len(qmpgrid)
        self.eigv = eigv
        self.eigvec = eigvec
        self.kmpgrid = kmpgrid
        self.qmpgrid = qmpgrid
        (self.kplusq_table, self.kminusq_table) = self.kmpgrid.get_kq_tables(self.qmpgrid)   
        try:
            self.q0index = self.qmpgrid.find_closest_kpoint([0.0,0.0,0.0])
        except ValueError:
            print('Warning! Q=0 index not found')
        self.dimbse = nk*nc*nv
        self.latdb = latdb
        # for now Transitions are the same as dipoles?
        self.T_table = T_table
        self.ctype = ctype
        self.ktype = ktype
        self.TD = TD #Tahm-Dancoff
        self.TBos = TBos
        self.method = method
        # consider to build occupations here in H2P with different occupation functions
        if (f_kn == None):
            self.f_kn = np.zeros((self.nk,self.nb),dtype=np.float128)
            self.f_kn[:,:nv] = 1.0
        else:
            self.f_kn = f_kn
        if(self.method=='model' and cpot is not None):
            print('\n Building H2P from model Coulomb potentials. Default is v2dt2\n')
            self.cpot = cpot
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
        if (self.nq == 0):        
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
                        H2P[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv] + (self.f_kn[ik,iv]-self.f_kn[ik,ic])*K_direct
                    else:
                        if (self.TD==True):
                            H2P[t,tp] = 0.0
                        else:
                            H2P[t,tp] = K_direct                
            return H2P
        else:
            H2P = np.zeros((self.nq,self.dimbse,self.dimbse),dtype=np.complex128)
            t0 = time()
            for iq in range(self.nq):
                for t in range(self.dimbse):
                    for tp in range(self.dimbse):
                        ik = self.T_table[t][0]
                        iv = self.T_table[t][1]
                        ic = self.T_table[t][2]
                        ikp = self.T_table[tp][0]
                        ivp = self.T_table[tp][1]
                        icp = self.T_table[tp][2]
                        # True power of Object oriented programming displayed in the next line
                        ikplusq = self.kplusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]+self.qmpgrid.k[iq]))
                        ikminusq = self.kminusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]-self.qmpgrid.k[iq]))
                        K_direct = self._getKdq(ik,iv,ic,ikp,ivp,icp,iq) 
                        K_Ex = self._getKEx(ik,iv,ic,ikp,ivp,icp,iq)
                        if (t==tp):
                            H2P[iq,t,tp] = self.eigv[ikplusq,ic]-self.eigv[ik,iv] + (self.f_kn[ik,iv]-self.f_kn[ikplusq,ic])*(K_direct - K_Ex)
                        else:
                            if (self.TD==True):
                                H2P[iq,t,tp] = 0.0
                            else:
                                H2P[iq,t,tp] = K_direct - K_Ex
            return H2P


    def _buildKernel(self, kernel):
        pass
        
    def _getexcitons(self, excitons):
        pass
        
    def solve_H2P(self):
        if(self.nq == 0):
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
        else:
            h2peigv = np.zeros((self.nq,self.dimbse), dtype=np.complex128)
            h2peigvec = np.zeros((self.nq,self.dimbse,self.dimbse),dtype=np.complex128)
            h2peigv_vck = np.zeros((self.nq,self.nv, self.nc, self.nk), dtype=np.complex128)
            h2peigvec_vck = np.zeros((self.nq,self.dimbse,self.nv,self.nc,self.nk),dtype=np.complex128)            
            for iq in range(0,self.nq):
                print(f'\nDiagonalizing the H2P matrix with dimensions: {self.dimbse} for q-point: {iq}\n')
                t0 = time()
                tmph2peigv = np.zeros((self.dimbse), dtype=np.complex128)
                tmph2peigvec = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
                tmph2peigv_vck = np.zeros((self.nv, self.nc, self.nk), dtype=np.complex128)
                tmph2peigvec_vck = np.zeros((self.dimbse,self.nv,self.nc,self.nk),dtype=np.complex128)
                (tmph2peigv, tmph2peigvec) = np.linalg.eigh(self.H2P[iq])
                #(self.h2peigv,self.h2peigvec) = sort_eig(self.h2peigv,self.h2peigvec)
                for t in range(self.dimbse):
                    ik, iv, ic = self.T_table[t]
                    tmph2peigvec_vck[:,iv, ic-self.nv, ik] = self.H2P[iq,:, t]   
                    tmph2peigv_vck[iv, ic-self.nv, ik] = tmph2peigv[t]
                
                h2peigv[iq] = tmph2peigv
                h2peigv_vck[iq] = tmph2peigv_vck        
                h2peigvec[iq] = tmph2peigvec
                h2peigvec_vck[iq] = tmph2peigvec_vck
            
            self.h2peigv = h2peigv
            self.h2peigv_vck = h2peigv_vck
            self.h2peigvec = h2peigvec
            self.h2peigvec_vck = h2peigvec_vck

            t1 = time()

            print(f'\n Diagonalization of H2P in {t1-t0:.3f} s')
    
    def _getKd(self,ik,iv,ic,ikp,ivp,icp):
        if (self.ktype =='IP'):
            K_direct = 0.0

            return K_direct
        
        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dt2(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:] )\
                        *np.vdot(self.eigvec[ik,ic],self.eigvec[ikp,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])
        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dk(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:] )\
                        *np.vdot(self.eigvec[ik,ic],self.eigvec[ikp,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])
        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_direct = self.cpot.vcoul(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ik,ic],self.eigvec[ikp,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])             
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ik,ic],self.eigvec[ikp,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])             
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ik,ic],self.eigvec[ikp,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])             
        return K_direct

    def _getKdq(self,ik,iv,ic,ikp,ivp,icp,iq):
        if (self.ktype =='IP'):
            K_direct = 0.0

            return K_direct

        ikplusq = self.kplusq_table[ik,iq]
        ikminusq = self.kminusq_table[ik,iq]      
        ikpplusq = self.kplusq_table[ikp,iq]
        ikpminusq = self.kminusq_table[ikp,iq]       

        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dt2(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ikpplusq,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])
        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dk(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ikpplusq,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])
        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_direct = self.cpot.vcoul(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ikpplusq,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])             
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ikpplusq,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])             
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.latdb.car_kpoints[ikp,:],self.latdb.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ikpplusq,icp])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ik,iv])             
        return K_direct
    
    def _getKEx(self,ik,iv,ic,ikp,ivp,icp,iq):
        if (self.ktype =='IP'):
            K_ex = 0.0

            return K_ex

        ikplusq = self.kplusq_table[ik,iq]
        ikminusq = self.kminusq_table[ik,iq]      
        ikpplusq = self.kplusq_table[ikp,iq]
        ikpminusq = self.kminusq_table[ikp,iq]         

        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_ex = self.cpot.v2dt2(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ik,iv])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ikpplusq,icp])
        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_ex = self.cpot.v2dk(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ik,iv])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ikpplusq,icp])
        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_ex = self.cpot.vcoul(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ik,iv])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ikpplusq,icp])
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')œ
            K_ex = self.cpot.v2dt(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ik,iv])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ikpplusq,icp])
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_ex = self.cpot.v2dt(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,ic],self.eigvec[ik,iv])*np.vdot(self.eigvec[ikp,ivp],self.eigvec[ikpplusq,icp])
        return K_ex
        
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
        if(self.nq != 0): 
            self.h2peigvec_vck=self.h2peigvec_vck[self.q0index]
            self.h2peigv_vck = self.h2peigv_vck[self.q0index]
            self.h2peigvec = self.h2peigvec[self.q0index]
            self.h2peigv = self.h2peigv[self.q0index]

        tb_dipoles = TB_dipoles(self.nk*self.nv*self.nc, self.nc, self.nv, self.nk, self.eigv,self.eigvec, eta, hlm, self.T_table,h2peigvec=self.h2peigvec_vck)
        # compute osc strength
        F_kcv = tb_dipoles.F_kcv
        # compute eps and pl
        f_pl = TB_occupations(self.eigv,Tel = 0, Tbos=self.TBos, Eb=self.h2peigv[0])._get_fkn( method='Boltz')
        pl = eps
        for ies, es in enumerate(w):
            for t in range(0,self.dimbse):
                eps[ies,:,:] += 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(self.h2peigv[t]-es)/(np.abs(es-self.h2peigv[t])**2+eta**2) \
                    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-self.h2peigv[t])**2+eta**2) 
                pl[ies,:,:] += f_pl * 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(self.h2peigv[t]-es)/(np.abs(es-self.h2peigv[t])**2+eta**2) \
                    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-self.h2peigv[t])**2+eta**2) 
        print('Excitonic Direct Ground state: ', self.h2peigv[0], ' [eV]')
        self.pl = pl
        # self.w = w
        # self.eps_0 = eps_0
        return w, eps
