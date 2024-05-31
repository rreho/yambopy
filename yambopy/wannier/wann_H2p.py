import numpy as np
from yambopy.wannier.wann_tb_mp import tb_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
from yambopy.wannier.wann_dipoles import TB_dipoles
from yambopy.wannier.wann_occupations import TB_occupations
from yambopy.dbs.bsekerneldb import *
from time import time



def process_file(args):
    idx, exc_db_file, data_dict = args
    # Unpacking data necessary for processing
    latdb, kernel_path, kpoints_indexes, HA2EV, BSE_table, kplusq_table, kminusq_table_yambo, eigv, f_kn = data_dict.values()

    yexc_atk = YamboExcitonDB.from_db_file(latdb, filename=exc_db_file)
    kernel_db = YamboBSEKernelDB.from_db_file(latdb, folder=f'{kernel_path}', Qpt=kpoints_indexes[idx]+1)
    K_ttp = kernel_db.kernel  # Assuming this returns a 2D array
    H2P_local = np.zeros((len(BSE_table), len(BSE_table)), dtype=np.complex128)

    for t in range(len(BSE_table)):
        ik, iv, ic = BSE_table[t]
        for tp in range(len(BSE_table)):
            ikp, ivp, icp = BSE_table[tp]
            ikplusq = kplusq_table[ik, kpoints_indexes[idx]]
            ikminusq = kminusq_table_yambo[ik, kpoints_indexes[idx]]
            ikpminusq = kminusq_table_yambo[ikp, kpoints_indexes[idx]]
            K = -(K_ttp[t, tp]) * HA2EV
            deltaE = eigv[ik, ic] - eigv[ikpminusq, iv] if (ik == ikp and icp == ic and ivp == iv) else 0.0
            occupation_diff = -f_kn[ikpminusq, ivp] + f_kn[ikp, icp]
            element_value = deltaE + occupation_diff * K
            H2P_local[t, tp] = element_value
    return idx, H2P_local


class H2P():
    '''Build the 2-particle resonant Hamiltonian H2P'''
    def __init__(self, nk, nb, nc, nv,eigv, eigvec,bse_nv, bse_nc, T_table, savedb, latdb, kmpgrid, qmpgrid,excitons=None, \
                  kernel_path=None, excitons_path=None,cpot=None,ctype='v2dt2',ktype='direct',bsetype='resonant', method='model',f_kn=None, \
                  TD=False,  TBos=300 , run_parallel=False): 
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
        self.nq_double = len(kmpgrid.k)
        self.nq = len(qmpgrid.k)
        self.eigv = eigv
        self.eigvec = eigvec
        self.kmpgrid = kmpgrid
        self.qmpgrid = qmpgrid
        self.kindices_table=self.kmpgrid.get_kindices_fromq(self.qmpgrid)
        (self.kplusq_table, self.kminusq_table) = self.kmpgrid.get_kq_tables(self.kmpgrid)  # the argument of get_kq_tables used to be self.qmpgrid. But for building the BSE hamiltonian we should not use the half-grid. To be tested for loop involving the q/2 hamiltonian  
        try:
            self.q0index = self.qmpgrid.find_closest_kpoint([0.0,0.0,0.0])
        except ValueError:
            print('Warning! Q=0 index not found')
        self.dimbse = bse_nv*bse_nc*nk
        self.savedb = savedb
        self.latdb = latdb
        self.offset_nv = self.savedb.nbandsv-self.nv  
        (self.kplusq_table_yambo, self.kminusq_table_yambo) = self.kmpgrid.get_kq_tables_yambo(self.savedb) # used in building BSE
        # for now Transitions are the same as dipoles?
        self.T_table = T_table
        self.BSE_table = self._get_BSE_table()
        self.ctype = ctype
        self.ktype = ktype
        self.TD = TD #Tahm-Dancoff
        self.TBos = TBos
        self.method = method
        self.run_parallel = True
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
                self.kernel_path = kernel_path
                self.excitons_path = excitons_path
            except  TypeError:
                print('Error Kernel is None or Path Not found')
            self.H2P = self._buildH2P()
        else:
            print('\nWarning! Kernel can be built only from Yambo database or model Coulomb potential\n')


    def _buildH2P(self):
        if self.run_parallel:
            import multiprocessing as mp
            import time
            pool = mp.Pool(mp.cpu_count())
            full_kpoints, kpoints_indexes, symmetry_indexes = self.savedb.expand_kpts()

            if self.nq_double == 1:
                H2P = np.zeros((self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = 'ndb.BS_diago_Q1'
            else:
                H2P = np.zeros((self.nq_double, self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = [f'ndb.BS_diago_Q{kpoints_indexes[iq] + 1}' for iq in range(self.nq_double)]

            exciton_db_files = [f'{self.excitons_path}/{suffix}' for suffix in np.atleast_1d(file_suffix)]
            t0 = time.time()

            # Prepare data to be passed
            data_dict = {
                'latdb': self.latdb,
                'kernel_path': self.kernel_path,
                'kpoints_indexes': kpoints_indexes,
                'HA2EV': HA2EV,
                'BSE_table': self.BSE_table,
                'kplusq_table': self.kplusq_table,
                'kminusq_table_yambo': self.kminusq_table_yambo,
                'eigv': self.eigv,
                'f_kn': self.f_kn
            }
            
            # Map with the necessary data tuple
            results = pool.map(process_file, [(idx, file, data_dict) for idx, file in enumerate(exciton_db_files)])
            for idx, result in results:
                if self.nq_double == 1:
                    H2P = result
                else:
                    H2P[idx] = result

            print(f"Hamiltonian matrix construction completed in {time.time() - t0:.2f} seconds.")
            pool.close()
            pool.join()
            return H2P

        else:

            # Expanded k-points and symmetry are prepared for operations that might need them
            full_kpoints, kpoints_indexes, symmetry_indexes = self.savedb.expand_kpts()

            # Pre-fetch all necessary data based on condition
            if self.nq_double == 1:
                H2P = np.zeros((self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = 'ndb.BS_diago_Q1'
            else:
                H2P = np.zeros((self.nq_double, self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = [f'ndb.BS_diago_Q{kpoints_indexes[iq] + 1}' for iq in range(self.nq_double)]

            # Common setup for databases (Yambo databases)
            exciton_db_files = [f'{self.excitons_path}/{suffix}' for suffix in np.atleast_1d(file_suffix)]
            # this is Yambo kernel, however I need an auxiliary kernel since the indices of c and v are different between BSE_table and BSE_table of YamboExcitonDB
            t0 = time()
            for idx, exc_db_file in enumerate(exciton_db_files):
                yexc_atk = YamboExcitonDB.from_db_file(self.latdb, filename=exc_db_file)
                v_band = np.min(yexc_atk.table[:, 1])
                c_band = np.max(yexc_atk.table[:, 2])
                kernel_db = YamboBSEKernelDB.from_db_file(self.latdb, folder=f'{self.kernel_path}',Qpt=kpoints_indexes[idx]+1)
                aux_t = np.lexsort((yexc_atk.table[:,2], yexc_atk.table[:,1],yexc_atk.table[:,0]))
                K_ttp = kernel_db.kernel[aux_t][:,aux_t]
                # Operations for matrix element calculations
                for t in range(self.dimbse):
                    ik, iv, ic = self.BSE_table[t]
                    for tp in range(self.dimbse):
                        ikp, ivp, icp = self.BSE_table[tp]
                        ikplusq = self.kplusq_table[ik, kpoints_indexes[idx]]
                        ikminusq = self.kminusq_table_yambo[ik, kpoints_indexes[idx]]
                        ikpminusq = self.kminusq_table_yambo[ikp,kpoints_indexes[idx]]
                        #print(ik, ikp, ikpminusq, idx, kpoints_indexes[idx])
                        #aux_t = kernel_db.get_kernel_indices_bands(yexc_atk, bands=[iv+self.offset_nv+1,ic+self.offset_nv+1],iq=ik+1)
                        #aux_tp = kernel_db.get_kernel_indices_bands(yexc_atk, bands=[ivp+self.offset_nv+1,icp+self.offset_nv+1],iq=ikpminusq+1)
                        
                        #K = -K_4D[ivp+self.offset_nv,icp+self.offset_nv,ikpminusq,ik]*HA2EV
                        K = -(K_ttp[t,tp])*HA2EV
                        #K = -K_ttp[aux_tp,aux_t]*HA2EV
                        #K=0
                        if (ik==ikp and icp==ic and ivp==iv):
                            deltaE = self.eigv[ik, ic] - self.eigv[ikpminusq, iv]
                        else:
                            deltaE = 0.0
                        occupation_diff = -self.f_kn[ikpminusq, ivp] + self.f_kn[ikp, icp]
                        element_value = deltaE + occupation_diff * K
                        if self.nq_double == 1:
                            H2P[t, tp] = element_value
                        else:
                            H2P[idx, t, tp] = element_value

            return H2P
    # def _buildH2P(self):
    #     # to-do fix inconsistencies with table
    #     # inconsistencies are in the k-points in mpgrid and lat.red_kpoints in Yambo
    #     full_kpoints, kpoints_indexes, symmetry_indexes=self.savedb.expand_kpts()
    #     if (self.nq_double ==1):
    #         H2P = np.zeros((self.dimbse,self.dimbse),dtype=np.complex128)
    #         t0 = time()
    #         yexc_atq = YamboExcitonDB.from_db_file(self.latdb, filename=f'{self.excitons_path}/ndb.BS_diago_Q1')
    #         kernel_atq = YamboBSEKernelDB.from_db_file(self.latdb, folder=f'{self.kernel_path}')                
    #         v_band = np.min(yexc_atq.table[:,1])
    #         c_band = np.min(yexc_atq.table[:,2])
    #         for t in range(self.dimbse):
    #             for tp in range(self.dimbse):
    #                 ik = self.BSE_table[t][0]
    #                 iv = self.BSE_table[t][1]
    #                 ic = self.BSE_table[t][2]
    #                 ikp = self.BSE_table[tp][0]
    #                 ivp = self.BSE_table[tp][1]
    #                 icp = self.BSE_table[tp][2]
    #                 # True power of Object oriented programming displayed in the next line
    #                 ikplusq = self.kplusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]+self.qmpgrid.k[iq]))
    #                 ikminusq = self.kminusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]-self.qmpgrid.k[iq]))
    #                 K = kernel_atq.kernel[t,tp]*HA2EV
    #                 if(t == tp):
    #                     H2P[t,tp] = self.eigv[ikminusq,ic]-self.eigv[ik,iv] + (self.f_kn[ik,iv]-self.f_kn[ikminusq,ic])*(K)
    #                     # if (self.TD==True):
    #                     #     H2P[iq,t,tp] = 0.0
    #                 else:
    #                     H2P[t,tp] =  (self.f_kn[ik,iv]-self.f_kn[ikminusq,ic])*(K)
    #         return H2P      
    #     else:
    #         H2P = np.zeros((self.nq_double,self.dimbse,self.dimbse),dtype=np.complex128)
    #         t0 = time()
    #         for iq in range(self.nq_double):
    #             yexc_atq = YamboExcitonDB.from_db_file(self.latdb, filename=f'{self.excitons_path}/ndb.BS_diago_Q{kpoints_indexes[iq]+1}')
    #             kernel_atq = YamboBSEKernelDB.from_db_file(self.latdb, folder=f'{self.kernel_path}')                
    #             v_band = np.min(yexc_atq.table[:,1])
    #             c_band = np.min(yexc_atq.table[:,2])                         
    #             for t in range(self.dimbse):
    #                 for tp in range(self.dimbse):
    #                     ik = self.BSE_table[t][0]
    #                     iv = self.BSE_table[t][1]
    #                     ic = self.BSE_table[t][2]
    #                     ikp = self.BSE_table[tp][0]
    #                     ivp = self.BSE_table[tp][1]
    #                     icp = self.BSE_table[tp][2]
    #                     # True power of Object oriented programming displayed in the next line
    #                     ikplusq = self.kplusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]+self.qmpgrid.k[iq]))
    #                     ikminusq = self.kminusq_table[ik,iq]#self.kmpgrid.find_closest_kpoint(self.kmpgrid.fold_into_bz(self.kmpgrid.k[ik]-self.qmpgrid.k[iq]))
    #                     K = kernel_atq.kernel[t,tp]*HA2EV
    #                     if(t == tp):
    #                         #print(ikminusq, iq, t ,tp ,H2P.shape, self.eigv.shape)
    #                         H2P[iq,t,tp] = self.eigv[ikminusq,ic]-self.eigv[ik,iv] + (self.f_kn[ik,iv]-self.f_kn[ikminusq,ic])*(K)
    #                         # if (self.TD==True):
    #                         #     H2P[iq,t,tp] = 0.0
    #                     else:
    #                         H2P[iq,t,tp] =   (self.f_kn[ik,ivp]-self.f_kn[ikminusq,icp])*(K)
    #         return H2P            

    
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
        if(self.nq_double == 1):
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
            h2peigv = np.zeros((self.nq_double,self.dimbse), dtype=np.complex128)
            h2peigvec = np.zeros((self.nq_double,self.dimbse,self.dimbse),dtype=np.complex128)
            h2peigv_vck = np.zeros((self.nq_double,self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
            h2peigvec_vck = np.zeros((self.nq_double,self.dimbse,self.bse_nv,self.bse_nc,self.nk),dtype=np.complex128) 
            deg_h2peigvec = np.array([])        
            for iq in range(0,self.nq_double):
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
        self.dipoles_bse = tb_dipoles.dipoles_bse
        self.dipoles = tb_dipoles.dipoles
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
    
    def get_eps_yambo(self, hlm, emin, emax, estep, eta):
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

        tb_dipoles = TB_dipoles(self.nc, self.nv, self.bse_nc, self.bse_nv, self.nk, self.eigv,self.eigvec, eta, hlm, self.T_table, self.BSE_table,h2peigvec=h2peigvec_vck, method='yambo')
        # compute osc strength
        F_kcv = tb_dipoles.F_kcv
        self.F_kcv = F_kcv
        self.dipoles_bse = tb_dipoles.dipoles_bse
        self.dipoles = tb_dipoles.dipoles
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

    def _get_exc_overlap_ttp(self, t, tp , iq, ikq, ib):
        '''Calculate M_SSp(Q,B) = \sum_{ccpvvpk}A^{*SQ}_{cvk}A^{*SpQ}_{cpvpk+B/2}*<u_{ck}|u_{cpk+B/2}><u_{vp-Q-B/2}|u_{vk-Q}>'''
        # missing the case of degenerate transitions. The summation seems to hold only for degenerate transitions.
        # Otherwise it's simply a product
        #6/12/23 after discussion with P.Melo I have to loop once again over all the transition indices
        Mssp_ttp = 0
        # Precompute indices and values that are reused
        ik, iv, ic = self.BSE_table[t]
        ikp, ivp, icp = self.BSE_table[tp]
        iqpb = self.kmpgrid.qpb_grid_table[iq, ib][1]
        
        for it in range(self.dimbse):
            # Precompute values used in the inner loop to reduce complexity
            eigvec_ic = self.eigvec[:, ic]
            eigvec_icp = self.eigvec[:, icp]
            eigvec_iv = self.eigvec[:, iv]
            eigvec_ivp = self.eigvec[:, ivp]

            for itp in range(self.dimbse):
                # Further decompose grid table access to simplify the formula
                ikmq = self.kmpgrid.kmq_grid_table[ik, iq][1]
                ikpbover2 = self.kmpgrid.kpbover2_grid_table[ik, ib][1]
                ikmqmbover2 = self.kmpgrid.kmqmbover2_grid_table[ik, iq, ib][1]

                # Decompose the complex arithmetic into more readable format
                conj_term = np.conjugate(self.h2peigvec_vck[ikq, t, self.bse_nv-self.nv+iv, ic-self.nv, ik])
                eigvec_term = self.h2peigvec_vck[iqpb, tp, self.bse_nv-self.nv+ivp, icp-self.nv, ikpbover2]
                dot_product1 = np.vdot(eigvec_ic, eigvec_icp)
                dot_product2 = np.vdot(eigvec_ivp, eigvec_iv)
                
                # Perform the summation
                Mssp_ttp += conj_term * eigvec_term * dot_product1 * dot_product2

        return Mssp_ttp        
        # Mssp_ttp = 0
        # for it in range(self.dimbse):
        #     for itp in range(self.dimbse):
        #         ik = self.BSE_table[t][0]
        #         iv = self.BSE_table[t][1] 
        #         ic = self.BSE_table[t][2] 
        #         ikp = self.BSE_table[tp][0]
        #         ivp = self.BSE_table[tp][1] 
        #         icp = self.BSE_table[tp][2] 
        #         iqpb = self.kmpgrid.qpb_grid_table[iq, ib][1]
        #         ikmq = self.kmpgrid.kmq_grid_table[ik,iq][1]
        #         ikpbover2 = self.kmpgrid.kpbover2_grid_table[ik, ib][1]
        #         ikmqmbover2 = self.kmpgrid.kmqmbover2_grid_table[ik, iq, ib][1]
        #         Mssp_ttp += np.conjugate(self.h2peigvec_vck[iq,t,self.bse_nv-self.nv+iv,ic-self.nv,ik])*self.h2peigvec_vck[iqpb,tp,self.bse_nv-self.nv+ivp,icp-self.nv, ikpbover2]*\
        #                         np.vdot(self.eigvec[ik,:, ic], self.eigvec[ikpbover2,:, icp])*np.vdot(self.eigvec[ikmqmbover2,:,ivp], self.eigvec[ikmq,:,iv]) 
        # return Mssp_ttp
    
    def get_exc_overlap(self, trange = [0], tprange = [0]):
        Mssp = np.zeros((len(trange), len(tprange),self.qmpgrid.nkpoints, self.qmpgrid.nnkpts), dtype=np.complex128)
        # here l stands for lambda, just to remember me that there is a small difference between lambda and transition index
        for il, l in enumerate(trange):
            for ilp, lp in enumerate(tprange):   
                for iq,ikq in enumerate(self.kindices_table):
                    for ib in range(self.qmpgrid.nnkpts):
                        Mssp[l,lp,iq, ib] = self._get_exc_overlap_ttp(l,lp,iq,ikq,ib)
        self.Mssp = Mssp   

    def _get_amn_ttp(self, t, tp, iq,ikq):
        ik = self.BSE_table[t][0]
        iv = self.BSE_table[t][1] 
        ic = self.BSE_table[t][2] 
        ikp = self.BSE_table[tp][0]
        ivp = self.BSE_table[tp][1] 
        icp = self.BSE_table[tp][2] 
        ikmq = self.kmpgrid.kmq_grid_table[ik,iq][1]
        Ammn_ttp = self.h2peigvec_vck[ikq,t, self.bse_nv-self.nv+iv, ic-self.nv,ik]*np.vdot(self.eigvec[ikmq,:,iv], self.eigvec[ik,:,ic])
        return Ammn_ttp

    def get_exc_amn(self, trange = [0], tprange = [0]):
        Amn = np.zeros((len(trange), len(tprange),self.qmpgrid.nkpoints), dtype=np.complex128)
        for it,t in enumerate(trange):
            for itp, tp in enumerate(tprange):
                for iq,ikq in enumerate(self.kindices_table):
                    Amn[t,tp, iq] = self._get_amn_ttp(t,tp,iq,ikq)        
        self.Amn = Amn

    def write_exc_overlap(self, seedname='wannier90_exc', trange=[0], tprange=[0]):
        if self.Mssp is None:
            self.get_exc_overlap(trange, tprange)

        from datetime import datetime

        # Using a context manager to handle file operations
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d at %H:%M:%S")
        output_lines = []

        # Preparing header and initial data
        output_lines.append(f'Created on {date_time_string}\n')
        output_lines.append(f'\t{len(trange)}\t{self.qmpgrid.nkpoints}\t{self.qmpgrid.nnkpts}\n')
        
        # Generate output for each point
        for iq,ikq in enumerate(self.kindices_table):
            for ib in range(self.qmpgrid.nnkpts):
                # Header for each block
                output_lines.append(f'\t{self.qmpgrid.qpb_grid_table[iq,ib][0]+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][1]+1}' +
                                    f'\t{self.qmpgrid.qpb_grid_table[iq,ib][2]}\t{self.qmpgrid.qpb_grid_table[iq,ib][3]}' +
                                    f'\t{self.qmpgrid.qpb_grid_table[iq,ib][4]}\n')
                # Data for each transition range
                for it, t in enumerate(trange):
                    for itp, tp in enumerate(tprange):
                        mssp_real = np.real(self.Mssp[tp, t, iq, ib])
                        mssp_imag = np.imag(self.Mssp[tp, t, iq, ib])
                        output_lines.append(f'\t{mssp_real:.14f}\t{mssp_imag:.14f}\n')

        # Writing all data at once
        with open(f'{seedname}.mmn', 'w') as f_out:
            f_out.writelines(output_lines)

        # current_datetime = datetime.now()
        # date_time_string = current_datetime.strftime("%Y-%m-%d at %H:%M:%S")
        # f_out = open(f'{seedname}.mmn', 'w')
        # f_out.write(f'Created on {date_time_string}\n')
        # f_out.write(f'\t{len(trange)}\t{self.qmpgrid.nkpoints}\t{self.qmpgrid.nnkpts}\n')        
        # for iq in range(self.qmpgrid.nkpoints):
        #     for ib in range(self.qmpgrid.nnkpts):
        #         # +1 is for Fortran counting
        #         f_out.write(f'\t{self.qmpgrid.qpb_grid_table[iq,ib][0]+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][1]+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][2]}\t{self.qmpgrid.qpb_grid_table[iq,ib][3]}\t{self.qmpgrid.qpb_grid_table[iq,ib][4]}\n')
        #         for it,t in enumerate(trange):
        #             for itp,tp in enumerate(tprange):
        #                 f_out.write(f'\t{np.real(self.Mssp[tp,t,iq,ib]):.14f}\t{np.imag(self.Mssp[tp,t,iq,ib]):.14f}\n')
        
        # f_out.close()

    def write_exc_eig(self, seedname='wannier90_exc', trange = [0]):
        exc_eig = np.zeros((len(trange), self.qmpgrid.nkpoints), dtype=complex)
        f_out = open(f'{seedname}.eig', 'w')
        for iq, ikq in enumerate(self.kindices_table):
            for it,t in enumerate(trange):
                f_out.write(f'\t{it+1}\t{iq+1}\t{np.real(self.h2peigv[ikq,it]):.13f}\n')
    
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
        for iq, q in enumerate(self.kindices_table):
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

    # def ensure_conjugate_symmetry(self,matrix):
    #     n = matrix.shape[0]
    #     for i in range(n):
    #         for j in range(i+1, n):
    #             if np.isclose(matrix[i, j].real, matrix[j, i].real) and np.isclose(matrix[i, j].imag, -matrix[j, i].imag):
    #                 matrix[j, i] = np.conjugate(matrix[i, j])
    #     return matrix