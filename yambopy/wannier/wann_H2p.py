import numpy as np
from yambopy.wannier.wann_tb_mp import tb_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
from yambopy.wannier.wann_utils import HA2EV, sort_eig
from yambopy.wannier.wann_dipoles import TB_dipoles
from yambopy.wannier.wann_occupations import TB_occupations
from yambopy.dbs.bsekerneldb import *
from time import time

class H2P():
    '''Build the 2-particle resonant Hamiltonian H2P'''
    def __init__(self, nk, nb, nc, nv,eigv, eigvec,bse_nv, bse_nc, T_table, latdb, kmpgrid, qmpgrid,excitons=None, \
                  kernel=None,cpot=None,ctype='v2dt2',ktype='direct',bsetype='resonant', method='model',f_kn=None, \
                  TD=False,  TBos=300): 
        '''Build H2P:
            bsetype = 'full' build H2P resonant + antiresonant + coupling
            bsetype = 'resonant' build H2p resonant
            TD is the Tahm-Dancoff which neglects the coupling
        '''
        self.nk = nk
        self.nb = nb
        self.nc = nc
        self.nv = nv
        self.bse_nv = bse_nv
        self.bse_nc = bse_nc
        self.nq = len(qmpgrid.k)
        self.eigv = eigv
        self.eigvec = eigvec
        self.kmpgrid = kmpgrid
        self.qmpgrid = qmpgrid
        (self.kplusq_table, self.kminusq_table) = self.kmpgrid.get_kq_tables(self.qmpgrid)   
        try:
            self.q0index = self.qmpgrid.find_closest_kpoint([0.0,0.0,0.0])
        except ValueError:
            print('Warning! Q=0 index not found')
        self.dimbse = bse_nv*bse_nc*nk
        self.latdb = latdb
        # for now Transitions are the same as dipoles?
        self.T_table = T_table
        self.BSE_table = self._get_BSE_table()
        self.ctype = ctype
        self.ktype = ktype
        self.TD = TD #Tahm-Dancoff
        self.TBos = TBos
        self.method = method
        self.Mssp = None
        self.Amn = None
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
                ik = self.BSE_table[t][0]
                iv = self.BSE_table[t][1]
                ic = self.BSE_table[t][2]
                ikp = self.BSE_table[tp][0]
                ivp = self.BSE_table[tp][1]
                icp = self.BSE_table[tp][2]
                #plus one needed in kernel for fortran counting
                # W_cvkc'v'k'
                if (t == tp):
                    H2P[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv] + self.kernel.get_kernel_value_bands(self.excitons,[iv+1,ic+1])[ik,ikp]*HA2EV 
                else:
                    H2P[t,tp] = self.kernel.get_kernel_value_bands(self.excitons,[iv+1,ic+1])[ik,ikp]*HA2EV
        return H2P
    
    def _buildH2P_fromcpot(self):
        'build resonant h2p from model coulomb potential'
        if (self.nq == 1):        
            H2P = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
            for t in range(self.dimbse):
                for tp in range(self.dimbse):
                    ik = self.BSE_table[t][0]
                    iv = self.BSE_table[t][1]
                    ic = self.BSE_table[t][2]
                    ikp = self.BSE_table[tp][0]
                    ivp = self.BSE_table[tp][1]
                    icp = self.BSE_table[tp][2]
                    K_direct = self._getKd(ik,iv,ic,ikp,ivp,icp)
                    if (t == tp ):
                        H2P[t,tp] = self.eigv[ik,ic]-self.eigv[ik,iv] - (self.f_kn[ik,iv]-self.f_kn[ik,ic])*K_direct # here -1 because (K_ex-K_direct) and K_ex = 0
                    else:
                        H2P[t,tp] = -(self.f_kn[ik,iv]-self.f_kn[ik,ic])*K_direct
                        #if (self.TD==True):
                        #    H2P[t,tp] = 0.0
                        #else:                                            
            return H2P
        else:
            H2P = np.zeros((self.nq,self.dimbse,self.dimbse),dtype=np.complex128)
            t0 = time()
            for iq in range(self.nq):
                for t in range(self.dimbse):
                    for tp in range(self.dimbse):
                        ik = self.BSE_table[t][0]
                        iv = self.BSE_table[t][1]
                        ic = self.BSE_table[t][2]
                        ikp = self.BSE_table[tp][0]
                        ivp = self.BSE_table[tp][1]
                        icp = self.BSE_table[tp][2]
                        # True power of Object oriented programming displayed in the next line
                        ikplusq = self.kplusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]+self.qmpgrid.k[iq]))
                        ikminusq = self.kminusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]-self.qmpgrid.k[iq]))
                        K_direct = self._getKdq(ik,iv,ic,ikp,ivp,icp,iq) 
                        K_Ex = self._getKEx(ik,iv,ic,ikp,ivp,icp,iq)
                        if(t == tp):
                            H2P[iq,t,tp] = self.eigv[ikplusq,ic]-self.eigv[ik,iv] + (self.f_kn[ik,iv]-self.f_kn[ikplusq,ic])*(K_Ex - K_direct)
                            # if (self.TD==True):
                            #     H2P[iq,t,tp] = 0.0
                        else:
                            H2P[iq,t,tp] =(self.f_kn[ik,iv]-self.f_kn[ikplusq,ic])*(K_Ex - K_direct)
            return H2P


    def _buildKernel(self, kernel):
        pass
        
    def _getexcitons(self, excitons):
        pass
        
    def solve_H2P(self):
        if(self.nq == 1):
            print(f'\nDiagonalizing the H2P matrix with dimensions: {self.dimbse}\n')
            t0 = time()
            self.h2peigv = np.zeros((self.dimbse), dtype=np.complex128)
            self.h2peigvec = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
            h2peigv_vck = np.zeros((self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
            h2peigvec_vck = np.zeros((self.dimbse,self.bse_nv,self.bse_nv,self.nk),dtype=np.complex128)
            (self.h2peigv, self.h2peigvec) = np.linalg.eigh(self.H2P)
            self.deg_h2peigvec = self.find_degenerate_eigenvalues(self.h2peigv, self.h2peigvec)
            #(self.h2peigv,self.h2peigvec) = sort_eig(self.h2peigv,self.h2peigvec)
            for t in range(self.dimbse):
                ik, iv, ic = self.BSE_table[t]
                h2peigvec_vck[:,self.bse_nv-self.nv+iv, ic-self.nv, ik] = self.h2peigvec[:,t]   
                h2peigv_vck[self.bse_nv-self.nv+iv, ic-self.nv, ik] = self.h2peigv[t]
            
            self.h2peigv_vck = h2peigv_vck        
            self.h2peigvec_vck = h2peigvec_vck
            self.deg_h2peigvec = deg_h2peigvec
            t1 = time()

            print(f'\n Diagonalization of H2P in {t1-t0:.3f} s')
        else:
            h2peigv = np.zeros((self.nq,self.dimbse), dtype=np.complex128)
            h2peigvec = np.zeros((self.nq,self.dimbse,self.dimbse),dtype=np.complex128)
            h2peigv_vck = np.zeros((self.nq,self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
            h2peigvec_vck = np.zeros((self.nq,self.dimbse,self.bse_nv,self.bse_nc,self.nk),dtype=np.complex128) 
            deg_h2peigvec = np.array([])        
            for iq in range(0,self.nq):
                print(f'\nDiagonalizing the H2P matrix with dimensions: {self.dimbse} for q-point: {iq}\n')
                t0 = time()
                tmph2peigv = np.zeros((self.dimbse), dtype=np.complex128)
                tmph2peigvec = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
                tmph2peigv_vck = np.zeros((self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
                tmph2peigvec_vck = np.zeros((self.dimbse,self.bse_nv,self.bse_nc,self.nk),dtype=np.complex128)
                (tmph2peigv, tmph2peigvec) = np.linalg.eigh(self.H2P[iq])
                deg_h2peigvec = np.append(deg_h2peigvec, self.find_degenerate_eigenvalues(tmph2peigv, tmph2peigvec))
                #(self.h2peigv,self.h2peigvec) = sort_eig(self.h2peigv,self.h2peigvec)
                for t in range(self.dimbse):
                    ik, iv, ic = self.BSE_table[t]
                    tmph2peigvec_vck[:,self.bse_nv-self.nv+iv, ic-self.nv, ik] = tmph2peigvec[:, t]   
                    tmph2peigv_vck[self.bse_nv-self.nv+iv, ic-self.nv, ik] = tmph2peigv[t]
                
                h2peigv[iq] = tmph2peigv
                h2peigv_vck[iq] = tmph2peigv_vck        
                h2peigvec[iq] = tmph2peigvec
                h2peigvec_vck[iq] = tmph2peigvec_vck
            
            self.h2peigv = h2peigv
            self.h2peigv_vck = h2peigv_vck
            self.h2peigvec = h2peigvec
            self.h2peigvec_vck = h2peigvec_vck
            self.deg_h2peigvec = deg_h2peigvec

            t1 = time()

            print(f'\n Diagonalization of H2P in {t1-t0:.3f} s')
    
    def _getKd(self,ik,iv,ic,ikp,ivp,icp):
        if (self.ktype =='IP'):
            K_direct = 0.0

            return K_direct
        
        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:] )\
                        *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])
        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dk(self.kmpgrid.car_kpoints[ikp,:],self.kmpgrid.car_kpoints[ik,:] )\
                        *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])
        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_direct = self.cpot.vcoul(self.kmpgrid.car_kpoints[ikp,:],self.kmpgrid.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.kmpgrid.car_kpoints[ikp,:],self.kmpgrid.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.kmpgrid.car_kpoints[ikp,:],self.kmpgrid.car_kpoints[ik,:])\
                        *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
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
            K_direct = self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:] )\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ikpplusq,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])
        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = self.cpot.v2dk(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:] )\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ikpplusq,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])
        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_direct = self.cpot.vcoul(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ikpplusq,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ikpplusq,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_direct = self.cpot.v2dt(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ikpplusq,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
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
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ik,:, iv])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ikpplusq,: ,icp])
        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_ex = self.cpot.v2dk(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ik,:, iv])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ikpplusq,:, icp])
        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_ex = self.cpot.vcoul(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ik,:, iv])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ikpplusq,:, icp])
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')œ
            K_ex = self.cpot.v2dt(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq,:, ic],self.eigvec[ik,:, iv])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ikpplusq,:, icp])
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_ex = self.cpot.v2dt(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.vdot(self.eigvec[ikplusq, : ,ic],self.eigvec[ik,:, iv])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ikpplusq,:, icp])
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
        if(self.nq != 1): 
            h2peigvec_vck=self.h2peigvec_vck[self.q0index]
            h2peigv_vck = self.h2peigv_vck[self.q0index]
            h2peigvec = self.h2peigvec[self.q0index]
            h2peigv = self.h2peigv[self.q0index]

        tb_dipoles = TB_dipoles(self.nc, self.nv, self.bse_nc, self.bse_nv, self.nk, self.eigv,self.eigvec, eta, hlm, self.T_table, self.BSE_table,h2peigvec=h2peigvec_vck)
        # compute osc strength
        F_kcv = tb_dipoles.F_kcv
        self.F_kcv = F_kcv
        # compute eps and pl
        #f_pl = TB_occupations(self.eigv,Tel = 0, Tbos=self.TBos, Eb=self.h2peigv[0])._get_fkn( method='Boltz')
        #pl = eps
        for ies, es in enumerate(w):
            for t in range(0,self.dimbse):
                eps[ies,:,:] += 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
                #pl[ies,:,:] += f_pl * 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                #    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
        print('Excitonic Direct Ground state: ', h2peigv[0], ' [eV]')
        #self.pl = pl
        # self.w = w
        # self.eps_0 = eps_0
        return w, eps

    def _get_exc_overlap_ttp(self, t, tp , iq, ib):
        '''Calculate M_SSp(Q,B) = \sum_{ccpvvpk}A^{*SQ}_{cvk}A^{*SpQ}_{cpvpk+B/2}*<u_{ck}|u_{cpk+B/2}><u_{vp-Q-B/2}|u_{vk-Q}>'''
        # missing the case of degenerate transitions. The summation seems to hold only for degenerate transitions.
        # Otherwise it's simply a product
        #6/12/23 after discussion with P.Melo I have to loop once again over all the transition indices
        Mssp_ttp = 0
        for it in range(self.dimbse):
            for itp in range(self.dimbse):
                ik = self.BSE_table[t][0]
                iv = self.BSE_table[t][1] 
                ic = self.BSE_table[t][2] 
                ikp = self.BSE_table[tp][0]
                ivp = self.BSE_table[tp][1] 
                icp = self.BSE_table[tp][2] 
                iqpb = self.qmpgrid.qpb_grid_table[iq, ib][1]
                ikmq = self.kmpgrid.kmq_grid_table[ik,iq][1]
                ikpbover2 = self.kmpgrid.kpbover2_grid_table[ik, ib][1]
                ikmqmbover2 = self.kmpgrid.kmqmbover2_grid_table[ik, iq, ib][1]
                Mssp_ttp += np.conjugate(self.h2peigvec_vck[iq,t,self.bse_nv-self.nv+iv,ic-self.nv,ik])*self.h2peigvec_vck[iqpb,tp,self.bse_nv-self.nv+ivp,icp-self.nv, ikpbover2]*\
                                np.vdot(self.eigvec[ik,:, ic], self.eigvec[ikpbover2,:, icp])*np.vdot(self.eigvec[ikmqmbover2,:,ivp], self.eigvec[ikmq,:,iv]) 
        return Mssp_ttp
    
    def get_exc_overlap(self, trange = [0], tprange = [0]):
        Mssp = np.zeros((len(trange), len(tprange),self.qmpgrid.nkpoints, self.qmpgrid.nnkpts), dtype=np.complex128)
        # here l stands for lambda, just to remember me that there is a small difference between lambda and transition index
        for il, l in enumerate(trange):
            for ilp, lp in enumerate(tprange):   
                for iq in range(self.qmpgrid.nkpoints):
                    for ib in range(self.qmpgrid.nnkpts):
                        Mssp[l,lp,iq, ib] = self._get_exc_overlap_ttp(l,lp,iq,ib)
        self.Mssp = Mssp   

    def _get_amn_ttp(self, t, tp, iq):
        ik = self.BSE_table[t][0]
        iv = self.BSE_table[t][1] 
        ic = self.BSE_table[t][2] 
        ikp = self.BSE_table[tp][0]
        ivp = self.BSE_table[tp][1] 
        icp = self.BSE_table[tp][2] 
        ikmq = self.kmpgrid.kmq_grid_table[ik,iq][1]
        Ammn_ttp = self.h2peigvec_vck[iq,t, self.bse_nv-self.nv+iv, ic-self.nv,ik]*np.vdot(self.eigvec[ikmq,:,iv], self.eigvec[ik,:,ic])
        return Ammn_ttp

    def get_exc_amn(self, trange = [0], tprange = [0]):
        Amn = np.zeros((len(trange), len(tprange),self.qmpgrid.nkpoints), dtype=np.complex128)
        for it,t in enumerate(trange):
            for itp, tp in enumerate(tprange):
                for iq in range(self.qmpgrid.nkpoints):
                    Amn[t,tp, iq] = self._get_amn_ttp(t,tp,iq)        
        self.Amn = Amn

    def write_exc_overlap(self,seedname='wannier90_exc', trange=[0], tprange=[0]):
        if (self.Mssp is None):
            self.get_exc_overlap(trange, tprange)

        from datetime import datetime
        
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d at %H:%M:%S")
        f_out = open(f'{seedname}.mmn', 'w')
        f_out.write(f'Created on {date_time_string}\n')
        f_out.write(f'\t{len(trange)}\t{self.qmpgrid.nkpoints}\t{self.qmpgrid.nnkpts}\n')        
        for iq in range(self.qmpgrid.nkpoints):
            for ib in range(self.qmpgrid.nnkpts):
                # +1 is for Fortran counting
                f_out.write(f'\t{self.qmpgrid.qpb_grid_table[iq,ib][0]+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][1]+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][2]}\t{self.qmpgrid.qpb_grid_table[iq,ib][3]}\t{self.qmpgrid.qpb_grid_table[iq,ib][4]}\n')
                for it,t in enumerate(trange):
                    for itp,tp in enumerate(tprange):
                        f_out.write(f'\t{np.real(self.Mssp[tp,t,iq,ib]):.14f}\t{np.imag(self.Mssp[tp,t,iq,ib]):.14f}\n')
        
        f_out.close()

    def write_exc_eig(self, seedname='wannier90_exc', trange = [0]):
        exc_eig = np.zeros((len(trange), self.qmpgrid.nkpoints), dtype=complex)
        f_out = open(f'{seedname}.eig', 'w')
        for iq, q in enumerate(self.qmpgrid.k):
            for it,t in enumerate(trange):
                f_out.write(f'\t{it+1}\t{iq+1}\t{np.real(self.h2peigv[iq,it]):.13f}\n')
    
    def write_exc_nnkp(self, seedname='wannier90_exc', trange = [0]):
        f_out = open(f'{seedname}.nnkp', 'w')
        f_out.write('begin nnkpts\n')
        f_out.write(f'\t{self.qmpgrid.nnkpts}\n')
        for iq, q in enumerate(self.qmpgrid.k):
            for ib in range(self.qmpgrid.nnkpts):
                iqpb = self.qmpgrid.qpb_grid_table[iq, ib][1]
                f_out.write(f'\t{iq+1}\t{iqpb+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][2]}\t{self.qmpgrid.qpb_grid_table[iq,ib][3]}\t{self.qmpgrid.qpb_grid_table[iq,ib][4]}\n')

    def write_exc_amn(self, seedname='wannier90_exc', trange = [0], tprange = [0]):
        if (self.Amn is None):
            self.get_exc_amn(trange, tprange)

        f_out = open(f'{seedname}.amn', 'w')

        from datetime import datetime

        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d at %H:%M:%S")
        f_out.write(f'Created on {date_time_string}\n')
        f_out.write(f'\t{len(trange)}\t{self.qmpgrid.nkpoints}\t{len(tprange)}\n') 
        for iq, q in enumerate(self.qmpgrid.k):
            for itp,tp in enumerate(tprange):
                for it, t in enumerate(trange):                
                    f_out.write(f'\t{it+1}\t{itp+1}\t{iq+1}\t{np.real(self.Amn[it,itp,iq])}\t\t{np.imag(self.Amn[it,itp,iq])}\n')

    def _get_BSE_table(self):
        ntransitions = self.nk*self.bse_nc*self.bse_nv
        BSE_table = np.zeros((ntransitions, 3),dtype=int)
        t_index = 0
        for ik in range(0,self.nk):
            for iv in range(0,self.bse_nv):
                for ic in range(0,self.bse_nc):
                        BSE_table[t_index] = [ik, self.nv-self.bse_nv+iv, self.nv+ic]
                        t_index += 1
        self.ntransitions = ntransitions
        return BSE_table
    
    def find_degenerate_eigenvalues(self, eigenvalues, eigenvectors, threshold=1e-3):
        degeneracies = {}
        for i, eigv1 in enumerate(eigenvalues):
            for j, eigv2 in enumerate(eigenvalues):
                if i < j and np.abs(eigv1 - eigv2) < threshold:
                    if i not in degeneracies:
                        degeneracies[i] = [i]
                    degeneracies[i].append(j)
        
        # Collect eigenvectors for each degenerate eigenvalue
        degenerate_eigenvectors = {key: eigenvectors[:, value] for key, value in degeneracies.items()}
        
        return degenerate_eigenvectors    
    