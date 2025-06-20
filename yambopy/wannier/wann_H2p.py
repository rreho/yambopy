import numpy as np
from yambopy.wannier.wann_asegrid import ase_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
from yambopy.wannier.wann_dipoles import TB_dipoles
from yambopy.wannier.wann_occupations import TB_occupations
from yambopy.dbs.bsekerneldb import *
from yambopy.dbs.electronsdb import *
from yambopy.dbs.latticedb import *
from yambopy.wannier.wann_io import AMN
from scipy.linalg.lapack import zheev
from time import time
import scipy
import gc

def process_file(args):
    idx, exc_db_file, data_dict = args
    # Unpacking data necessary for processing
    latdb, kernel_path, kpoints_indexes, HA2EV, BSE_table, kplusq_table, kminusq_table_yambo, eigv, f_kn = data_dict.values()

    yexc_atk = YamboExcitonDB.from_db_file(latdb, filename=exc_db_file)
    kernel_db = YamboBSEKernelDB.from_db_file(latdb, folder=f'{kernel_path}', Qpt=kpoints_indexes[idx]+1)
    aux_t = np.lexsort((yexc_atk.table[:,2], yexc_atk.table[:,1],yexc_atk.table[:,0]))
    K_ttp = kernel_db.kernel[aux_t][:,aux_t]  
    H2P_local = np.zeros((len(BSE_table), len(BSE_table)), dtype=np.complex128)

    del kernel_db, yexc_atk  # Free memory as early as possible

    BSE_table = np.array(BSE_table)
    ik = BSE_table[:, 0]
    iv = BSE_table[:, 1]
    ic = BSE_table[:, 2]

    ikp = BSE_table[:, 0]
    ivp = BSE_table[:, 1]
    icp = BSE_table[:, 2]

    ikplusq = kplusq_table[ik, kpoints_indexes[idx],1]
    ikminusq = kminusq_table_yambo[ik, kpoints_indexes[idx],1]
    ikpminusq = kminusq_table_yambo[ikp, kpoints_indexes[idx],1]

    # Ensure deltaE is diagonal
    deltaE = np.zeros((len(BSE_table), len(BSE_table)),dtype=np.complex128)
    diag_indices = np.arange(len(BSE_table))
    mask = (ik == ikp) & (ic == icp) & (iv == ivp)
    deltaE[diag_indices, diag_indices] = np.where(mask, eigv[ik, ic] - eigv[ikpminusq, iv], 0.0)

    occupation_diff = -f_kn[ikpminusq, ivp] + f_kn[ikp, icp]
    del ik, iv, ic, ikp, ivp, icp, ikplusq, ikminusq, ikpminusq  # Free memory
    K = -(K_ttp[np.arange(BSE_table.shape[0])[:, None], np.arange(BSE_table.shape[0])[None, :]]) * HA2EV

    # Ensure deltaE is diagonal in (t, tp)
    H2P_local = deltaE + occupation_diff[:, None] * K

    del deltaE, occupation_diff, K, K_ttp  # Free memory
    gc.collect()  # Force garbage collection

    return idx, H2P_local    

    # for t in range(len(BSE_table)):
    #     ik, iv, ic = BSE_table[t]
    #     for tp in range(len(BSE_table)):
    #         ikp, ivp, icp = BSE_table[tp]
    #         ikplusq = kplusq_table[ik, kpoints_indexes[idx]]
    #         ikminusq = kminusq_table_yambo[ik, kpoints_indexes[idx]]
    #         ikpminusq = kminusq_table_yambo[ikp, kpoints_indexes[idx]]
    #         K = -(K_ttp[t, tp]) * HA2EV
    #         deltaE = eigv[ik, ic] - eigv[ikpminusq, iv] if (ik == ikp and icp == ic and ivp == iv) else 0.0
    #         occupation_diff = -f_kn[ikpminusq, ivp] + f_kn[ikp, icp]
    #         element_value = deltaE + occupation_diff * K
    #         H2P_local[t, tp] = element_value
    # return idx, H2P_local

class FakeLatticeObject():
    '''should make the YamboLatticeDB class actually more lightweight and
    allow for other ways of initiliazing with only the unit cell for instance because we don't need most things.
    We only need: 
    .lat_vol
    .alat

    YamboBSEKernelDB class only uses it when it needs kernel values per bands, and in that case it only need nkpoints
    
    '''
    def __init__(self, model,win_file=None):
        self.alat = model.uc * 1/0.52917720859      # this results in minor difference, do we really want the values from yambo?
        self.lat = self.alat
        self.lat_vol = np.prod(np.diag(self.alat))  # difference becomes bigger
        self.rlat = model.reciprocal_lattice
        self.rlat_vol = np.prod(np.diag(self.rlat))
        if win_file:
            from .wann_io import WIN
            self.red_atomic_positions, self.atomic_numbers =  WIN(win_file).read_positions()

class H2P():
    '''Build the 2-particle resonant Hamiltonian H2P
        Easy to use only requires the model as input. 
    '''
    def __init__(self, model, electronsdb_path, kmpgrid, qmpgrid, bse_nv=1, bse_nc=1, kernel_path=None, excitons_path=None,cpot=None, \
                 ctype='v2dt2',ktype='direct',bsetype='resonant', method='model',f_kn=None, f_qn = None,\
                 TD=False, run_parallel=False,dimslepc=100,gammaonly=False,nproc=8,eta=0.01):
    
    # nk, nb, nc, nv,eigv, eigvec, bse_nv, bse_nc, T_table, electronsdb, kmpgrid, qmpgrid,excitons=None, \
    #               kernel_path=None, excitons_path=None,cpot=None,ctype='v2dt2',ktype='direct',bsetype='resonant', method='model',f_kn=None, \
    #               TD=False,  TBos=300 , run_parallel=False): 
        '''Build H2P:
            bsetype = 'full' build H2P resonant + antiresonant + coupling
            bsetype = 'resonant' build H2p resonant
            TD is the Tahm-Dancoff which neglects the coupling
        '''
        self.model = model
        self.nk = model.nk
        self.nb = model.nb
        self.nc = model.nc
        self.nv = model.nv
        self.bse_nv = bse_nv
        self.bse_nc = bse_nc
        self.kmpgrid = kmpgrid
        self.qmpgrid = qmpgrid
        self.nq = len(qmpgrid.k)
        self.eigv = model.eigv
        self.eigvec = model.eigvec
        self.correct_neighbours()
        self.gammaonly=gammaonly
        self.eta = eta
        if(self.gammaonly):
            self.nq_double = 1
        else:
            self.nq_double = len(self.qmpgrid.k)
        self.kindices_table=self.kmpgrid.get_kindices_fromq(self.qmpgrid) # get a q point in the qgrid and return index the the q point in the k grid
        self.qindices_table={v: i for i, v in enumerate(self.kindices_table)} # get a q point in the qgrid expressed in the k grid and return index of the qpoint in the qgrid
        try:
            self.q0index = self.qmpgrid.find_closest_kpoint([0.0,0.0,0.0])
        except ValueError:
            print('Warning! Q=0 index not found')
        self.dimbse = self.bse_nv*self.bse_nc*self.nq
        if electronsdb_path:
            self.electronsdb_path = electronsdb_path
            self.electronsdb = YamboElectronsDB.from_db_file(folder=f'{electronsdb_path}', Expand=True)
            self.latdb = YamboLatticeDB.from_db_file(folder=f'{electronsdb_path}', Expand=True)
        else:
            self.electronsdb_path = None    # required when initializing ExcitonBands in pp
            self.electronsdb = FakeLatticeObject(model)
            self.latdb = FakeLatticeObject(model)
        self.offset_nv = self.nv-self.bse_nv
        self.T_table = model.T_table
        self.BSE_table = self._get_BSE_table()
        self.ktype = ktype
        self.TD = TD #Tahm-Dancoff
        self.method = method
        self.run_parallel = run_parallel
        self.Mssp = None
        self.Amn = None
        self.skip_diago = False
        self.nproc = nproc
        # consider to build occupations here in H2P with different occupation functions
        if not hasattr(self,'f_kn'):
            self.f_kn = np.zeros((self.nk,self.nb),dtype=np.float64)
            self.f_kn[:,:self.nv] = 1.0
        else:
            self.f_kn = f_kn
        if not hasattr(self,'f_qn'):
            self.f_qn = np.zeros((self.nq_double,self.nb),dtype = np.float64)
            self.f_qn[:,:self.nv] = 1.0
        else:
            self.f_qn = f_qn
        if(self.method=='model' and cpot is not None):
            self.ctype = ctype
            (self.kplusq_table, self.kminusq_table) = self.kmpgrid.get_kq_tables(self.qmpgrid)  # the argument of get_kq_tables used to be self.qmpgrid. But for building the BSE hamiltonian we should not use the half-grid. To be tested for loop involving the q/2 hamiltonian  
            (self.qplusk_table, self.qminusk_table) = self.qmpgrid.get_kq_tables(self.kmpgrid, sign='-')  # minus sign to have k-q  
            #(self.kplusq_table_yambo, self.kminusq_table_yambo) = self.kmpgrid.get_kq_tables_yambo(self.electronsdb) # used in building BSE
            print(f'\n Building H2P from model Coulomb potentials {self.ctype}\n')
            self.cpot = cpot
            self.H2P = self._buildH2P_fromcpot()
        elif(self.method=='kernel' and cpot is None):
            (self.kplusq_table, self.kminusq_table) = self.kmpgrid.get_kq_tables(self.qmpgrid)
            (self.qplusk_table, self.qminusk_table) = self.qmpgrid.get_kq_tables(self.kmpgrid, sign='-')  # minus sign to have k-q  
            (self.kplusq_table_yambo, self.kminusq_table_yambo) = self.kmpgrid.get_kq_tables_yambo(self.electronsdb) # used in building BSE
            print('\n Building H2P from model YamboKernelDB\n')
            #Remember that the BSEtable in Yambopy start counting from 1 and not to 0
            try:
                self.kernel_path = kernel_path
                if not excitons_path:
                    self.excitons_path = kernel_path
                else:
                    self.excitons_path = excitons_path
            except  TypeError:
                print('Error Kernel is None or Path Not found')
            self.H2P = self._buildH2P()

        elif(self.method=='skip-diago' and cpot is None):
            self.excitons_path = excitons_path
            print('Method skip-diago running only for post-processing of wannier exc data: Remember to set dimslepc')
            self.skip_diago = True
            self.dimslepc=dimslepc
            (self.h2peigv, self.h2peigvec,self.h2peigv_vck, self.h2peigvec_vck) = self._buildH2Peigv()
        else:
            print('\nWarning! Kernel can be built only from Yambo database or model Coulomb potential\n')

    def correct_neighbours(self):
        # correct the eigv sorting wrt neighbours
        sort_idx = self.kmpgrid.sort_idx
        self.eigv = np.take_along_axis(self.eigv, sort_idx[:,:], axis=1)
        tmp = np.take_along_axis(self.eigvec, sort_idx[:, :, None], axis=1)
        self.eigvec = np.take_along_axis(tmp, sort_idx[:, None, :], axis=2)

    def _buildH2P(self):
        if self.run_parallel:
            import multiprocessing as mp
            cpucount= mp.cpu_count()
            print(f"CPU count involved in H2P loading pool: {cpucount}")
            pool = mp.Pool(self.nproc)
            full_kpoints, kpoints_indexes, symmetry_indexes = self.electronsdb.iku_kpoints, self.electronsdb.kpoints_indexes, self.electronsdb.symmetry_indexes
            
            self.nq_double = len(full_kpoints)
            # full_kpoints, kpoints_indexes, symmetry_indexes = self.electronsdb.expand_kpts()
            if self.nq_double == 1:
                H2P = np.zeros((self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = 'ndb.BS_diago_Q1'
            else:
                H2P = np.zeros((self.nq_double, self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = [f'ndb.BS_diago_Q{kpoints_indexes[iq] + 1}' for iq in range(self.nq_double)]

            exciton_db_files = [f'{self.excitons_path}/{suffix}' for suffix in np.atleast_1d(file_suffix)]
            t0 = time()

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
            del results  # Free up memory by deleting the results
            gc.collect()  # Force garbage collection
            print(f"Hamiltonian matrix construction completed in {time() - t0:.2f} seconds.")
            pool.close()
            pool.join()
            return H2P

        else:
            # Expanded k-points and symmetry are prepared for operations that might need them
            full_kpoints, kpoints_indexes, symmetry_indexes = self.electronsdb.iku_kpoints, self.electronsdb.kpoints_indexes, self.electronsdb.symmetry_indexes
            self.nq_double = len(full_kpoints)
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
                BSE_table = np.array(self.BSE_table)
                ik = BSE_table[:, 0]
                iv = BSE_table[:, 1]
                ic = BSE_table[:, 2]

                ikp = BSE_table[:, 0]
                ivp = BSE_table[:, 1]
                icp = BSE_table[:, 2]

                ikplusq = self.kplusq_table_yambo[ik, kpoints_indexes[idx],1]
                ikminusq = self.kminusq_table_yambo[ik, kpoints_indexes[idx],1]
                ikpminusq = self.kminusq_table_yambo[ikp, kpoints_indexes[idx],1]

                # Ensure deltaE is diagonal
                deltaE = np.zeros((self.dimbse, self.dimbse),dtype=np.complex128)
                diag_indices = np.arange(self.dimbse)
                mask = (ik == ikp) & (ic == icp) & (iv == ivp)
                deltaE[diag_indices, diag_indices] = np.where(mask, self.eigv[ik, ic] - self.eigv[ikpminusq, iv], 0.0)

                occupation_diff = -self.f_kn[ikpminusq, ivp] + self.f_kn[ikp, icp]

                # Refactored K calculation
                K = -(K_ttp[np.arange(self.dimbse)[:, None], np.arange(self.dimbse)[None, :]]) * HA2EV

                element_value = deltaE + occupation_diff[:, None] * K
              
                if self.nq_double == 1:
                    H2P[:, :] = element_value
                else:
                    H2P[idx, :, :] = element_value
                del element_value  # Free up memory by deleting the results
                gc.collect()  # Force garbage collection
            print(f"Hamiltonian matrix construction completed in {time() - t0:.2f} seconds.")
            return H2P              
        
    def _buildH2Peigv(self):
        if self.skip_diago:
            H2P = None
            full_kpoints, kpoints_indexes, symmetry_indexes = self.electronsdb.iku_kpoints, self.electronsdb.kpoints_indexes, self.electronsdb.symmetry_indexes
            # self.qgrid_toibzk = self.electronsdb.kpoints_indexes[self.kindices_table[:]]
            # self.ibzk_toqgrid={v: i for i, v in enumerate(self.qgrid_toibzk)}

            if self.nq_double == 1:
                H2P = np.zeros((self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = 'ndb.BS_diago_Q1'
            elif self.nq_double == self.nq:
                print("Using the Full Brillouin zone.")
                H2P = np.zeros((self.nq_double, self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = [f'ndb.BS_diago_Q{iq + 1}' for iq in range(self.nq_double)]
            else:
                print("Rotating (improperly the exciton wavefunction.")
                H2P = np.zeros((self.nq_double, self.dimbse, self.dimbse), dtype=np.complex128)
                file_suffix = [f'ndb.BS_diago_Q{self.latdb.kpoints_indexes[iq] + 1}' for iq in range(self.nq_double)]

            exciton_db_files = [f'{self.excitons_path}/{suffix}' for suffix in np.atleast_1d(file_suffix)]
            h2peigv_vck = np.zeros((self.nq_double, self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
            h2peigvec_vck = np.zeros((self.nq_double, self.dimslepc, self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
            h2peigv = np.zeros((self.nq_double, self.dimslepc), dtype=np.complex128)
            h2peigvec = np.zeros((self.nq_double, self.dimslepc, self.dimbse), dtype=np.complex128)
            t0 = time()

            for idx, exc_db_file in enumerate(exciton_db_files):
                yexc_atk = YamboExcitonDB.from_db_file(self.latdb, filename=exc_db_file)
                aux_t = np.lexsort((yexc_atk.table[:, 2], yexc_atk.table[:, 1], yexc_atk.table[:, 0]))  # c,v,k
                # Create an array to store the inverse mapping
                inverse_aux_t = np.empty_like(aux_t)
                # Populate the inverse mapping
                inverse_aux_t[aux_t] = np.arange(aux_t.size)
                self.inverse_aux_t = inverse_aux_t
                tmph2peigvec = yexc_atk.eigenvectors.filled(0).copy()
                tmph2peigv = yexc_atk.eigenvalues.filled(0).copy()

                BSE_table = np.array(self.BSE_table)
                ik = BSE_table[inverse_aux_t, 0]
                iv = BSE_table[inverse_aux_t, 1]
                ic = BSE_table[inverse_aux_t, 2]
                


                # Broadcasting and advanced indexing
                inverse_aux_t_slepc = inverse_aux_t[:self.dimslepc]
                h2peigv[idx, :] = tmph2peigv
                h2peigvec[idx, :, :] = tmph2peigvec[:self.dimslepc, :]

                ik_t = ik[:self.dimslepc]
                iv_t = iv[:self.dimslepc]
                ic_t = ic[:self.dimslepc]
                #this should be called with inverse_aux_t
                h2peigv_vck[idx, self.bse_nv - self.nv + iv_t, ic_t - self.nv, ik_t] = tmph2peigv[:self.dimslepc]

                ikp_indices = BSE_table[inverse_aux_t, 0]
                ivp_indices = BSE_table[inverse_aux_t, 1]
                icp_indices = BSE_table[inverse_aux_t, 2]
                #tmph2peigvec = tmph2peigvec.reshape((1, 100, 648))
                tmp_t = np.arange(0,self.dimslepc)
                #first t index should be called normally, second with inverse_aux_t
                h2peigvec_vck[idx, tmp_t[:,None], self.bse_nv - self.nv + ivp_indices[None,:], icp_indices[None,:] - self.nv, ikp_indices[None,:]] = tmph2peigvec[:, :]

            self.BSE_table = self.BSE_table[self.inverse_aux_t]

            self.H2P = H2P
            print(f"Reading excitonic eigenvalues and eigenvectors in {time() - t0:.2f} seconds.")
            return h2peigv, h2peigvec, h2peigv_vck, h2peigvec_vck
        else:
            print('Error: skip_diago is false')         

    def _buildH2P_fromcpot(self):
        'build resonant h2p from model coulomb potential'

        H2P = np.zeros((self.nq_double, self.dimbse, self.dimbse), dtype=np.complex128)
        print('initialize buildh2p from cpot')
        t0 = time()

        # Precompute kplusq and kminusq tables
        # ikpminusq = self.kplusq_table[:, :, 1]
        ikminusq = self.kminusq_table[:, :, 1]
        ikminusgamma = self.kminusq_table[:, :, 0]



        K_direct, K_Ex = self._getKdq()       #sorry

        gc.collect()
        K_diff = K_direct - K_Ex[:,np.newaxis,:]
        f_kmqn = np.tile(self.f_qn[None, :, :], (self.nk, 1, 1))
        # f_kmqn = self.f_qn.tile().reshape(self.nk, self.nq_double, self.nb)
        # f_diff = self.f_kn[ikminusq][:,self.BSE_table[:,0],:][:,:,self.BSE_table[:,1]]
        # f_diff -=f_kmqn[ikminusgamma][:,self.BSE_table[:,0],:][:,:,self.BSE_table[:,2]] 

        f_diff = (self.f_kn[self.BSE_table[:,0],:][:,self.BSE_table[:,1]][None,:,:]-f_kmqn[self.BSE_table[:,0],:,:][:,:,self.BSE_table[:,2]].swapaxes(1,0))
        del K_Ex
        gc.collect()
        
        H2P = f_diff * K_diff
        del K_diff, f_diff
        gc.collect()
        
        result = self.eigv[ikminusq[self.BSE_table[:, 0]], self.BSE_table[:, 1][:, None]].T  # Shape: (nqpoints, ntransitions)
        eigv_diff = self.eigv[self.BSE_table[:,0],self.BSE_table[:,2]] - result
        del result
        gc.collect()
        self.eigv_diff_ttp = eigv_diff
        del eigv_diff
        gc.collect()
        
        diag = np.einsum('ij,ki->kij', np.eye(self.dimbse), self.eigv_diff_ttp)  # when t ==tp
        H2P += diag
        del diag 
        gc.collect()
        print(f'Completed in {time() - t0} seconds')
        return H2P

    def _buildKernel(self, kernel):
        pass
        
    def _getexcitons(self, excitons):
        pass
        
    def solve_H2P(self):

        h2peigv = np.zeros((self.nq_double,self.dimbse), dtype=np.complex128)
        h2peigvec = np.zeros((self.nq_double,self.dimbse,self.dimbse),dtype=np.complex128)
        h2peigv_vck = np.zeros((self.nq_double,self.bse_nv, self.bse_nc, self.nk), dtype=np.complex128)
        h2peigvec_vck = np.zeros((self.nq_double,self.dimbse,self.bse_nv,self.bse_nc,self.nk),dtype=np.complex128) 
        deg_h2peigvec = np.array([])        
        
        print(f'\nDiagonalizing the H2P matrix with dimensions: {self.H2P.shape} \n')
        t0 = time()
        for iq in range(0,self.nq_double):

            (h2peigv[iq],h2peigvec[iq]) = scipy.linalg.eigh(self.H2P[iq])

            h2peigvec_vck[iq][:, self.bse_nv - self.nv+self.BSE_table[:,1], self.BSE_table[:,2]-self.nv, self.BSE_table[:,0]] = h2peigvec[iq].T
            h2peigv_vck[iq][self.bse_nv - self.nv + self.BSE_table[:,1], self.BSE_table[:,2] - self.nv, self.BSE_table[:,0]] = h2peigv[iq]
        

        self.h2peigv = h2peigv
        self.h2peigv_vck = h2peigv_vck
        self.h2peigvec = h2peigvec
        self.h2peigvec_vck = h2peigvec_vck

        t1 = time()

        print(f'\n Diagonalization of H2P in {t1-t0:.3f} s')
    
    def _getKd(self,ik,iv,ic,ikp,ivp,icp):
        if (self.ktype =='IP'):
            K_direct = 0.0
            print('ciao')
            return K_direct
        
        elif (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_direct = +self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:] )\
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
            # K_direct = self.cpot.v2drk(self.kmpgrid.car_kpoints[ikp,:],self.kmpgrid.car_kpoints[ik,:])\
                        # *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikp,:, ivp],self.eigvec[ik,:, iv])             
            ikminusq = self.kminusq_table[ik, 1]
            eigc = self.eigvec[self.BSE_table[:,0], :, self.BSE_table[:,2]][:,np.newaxis,:]   # conduction bands
            eigcp = self.eigvec[self.BSE_table[:,0], :, self.BSE_table[:,2]][np.newaxis,:,:]   # conduction bands prime
            eigv = self.eigvec[ikminusq, :, :][self.BSE_table[:,0],:,:,self.BSE_table[:,1]][:,np.newaxis,:,:]  # Valence bands of ikminusq
            eigvp = self.eigvec[ikminusq, :, :][self.BSE_table[:,0],:,:,self.BSE_table[:,1]][np.newaxis,:,:,:]  # Valence bands prime of ikminusq
            
            
            dotc = np.einsum('ijk,ijk->ij',np.conjugate(eigc), eigcp)
            dotv = np.einsum('ijkl,ijkl->kij',np.conjugate(eigvp), eigv)

            del eigc, eigcp, eigv, eigvp
            gc.collect()

            v2drk_array = self.cpot.v2drk(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)
            
            K_direct = v2drk_array[self.BSE_table[:,0],][:,self.BSE_table[:,0]] * dotc * dotv
            del dotc, dotv, v2drk_array
            gc.collect()

            return K_direct

    def _getKdq(self):
        if (self.ktype =='IP'):
            K_direct = 0.0
            print('ciao')

            return K_direct
        cpot_array = None

        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            # Ensure inputs are NumPy arrays
            #K_direct = self.cpot.v2dt2(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
            #   *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])
            cpot_array = self.cpot.v2dt2(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)

        
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
                        # K_direct = self.cpot.v2dk(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:] )\
                        # *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])
            cpot_array = self.cpot.v2dk(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)

        
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            cpot_array = self.cpot.vcoul(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)

        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            cpot_array = self.cpot.v2dt(self.kmpgrid.car_kpoints, self.kmpgrid.car_kpoints)


        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            # K_direct = self.cpot.v2drk(self.kmpgrid.car_kpoints[ik,:],self.kmpgrid.car_kpoints[ikp,:])\
                        # *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikp,:, icp])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikminusq,:, iv])            
            # K_ex = self.cpot.v2drk(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        # *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikminusq,:, iv])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikp,: ,icp])
            cpot_array = self.cpot.v2drk(self.kmpgrid.car_kpoints,self.kmpgrid.car_kpoints)

        ikminusq = self.kminusq_table[:, :, 1]
        eigc = self.eigvec[self.BSE_table[:,0], :, self.BSE_table[:,2]][:,np.newaxis,:]   # conduction bands
        eigcp = self.eigvec[self.BSE_table[:,0], :, self.BSE_table[:,2]][np.newaxis,:,:]   # conduction bands prime
        eigv = self.eigvec[ikminusq, :, :][self.BSE_table[:,0],:,:,self.BSE_table[:,1]][:,np.newaxis,:,:]  # Valence bands of ikminusq
        eigvp = self.eigvec[ikminusq, :, :][self.BSE_table[:,0],:,:,self.BSE_table[:,1]][np.newaxis,:,:,:]  # Valence bands prime of ikminusq

        self.eigvecc_t = eigc[:,0,:]
        self.eigvecv_t = eigv[:,0,0,:]

        dotc = np.einsum('ijk,ijk->ij',np.conjugate(eigc), eigcp)
        dotv = np.einsum('ijkl,ijkl->kij',np.conjugate(eigv), eigvp)
        K_direct = cpot_array[self.BSE_table[:,0],][:,self.BSE_table[:,0]] * dotc * dotv
        del dotc, dotv
        gc.collect()
        
        if self.nq_double == 1:
            return K_direct, np.array([[0.0]])
        
        dotc2 = np.einsum('ijk,jilk->li',np.conjugate(eigc), eigvp)
        dotv2 = np.einsum('ijkl,jil->ki',np.conjugate(eigv), eigcp)
        
        del eigc, eigcp, eigv, eigvp  
        gc.collect()


        K_Ex = cpot_array[0][self.BSE_table[:,0]] * dotc2 * dotv2
        del cpot_array
        gc.collect()

        return K_direct, K_Ex
    
    def _getKEx(self,ik,iv,ic,ikp,ivp,icp,iq):
        if (self.ktype =='IP'):
            K_ex = 0.0
            print('ciao')

            return K_ex

        ikplusq = self.kplusq_table[ik,iq,1]
        ikminusq = self.kminusq_table[ik,iq,1]      
        ikpplusq = self.kplusq_table[ikp,iq,1]
        ikpminusq = self.kminusq_table[ikp,iq,1]         

        if (self.ctype=='v2dt2'):
            #print('\n Kernel built from v2dt2 Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            # K_ex = self.cpot.v2dt2(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        # *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikminusq,:, iv])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikp,: ,icp])
            kpt1 = self.kmpgrid.car_kpoints
            kpt2 = self.kmpgrid.car_kpoints

            kpt1_broadcasted = kpt1[:, np.newaxis, :]  # Shape (N1, 1, 3)
            kpt2_broadcasted = kpt2[np.newaxis, :, :]  # Shape (1, N2, 3)
                
            # Compute modk for all kpt1 and kpt2 pairs
            modk = np.linalg.norm(kpt1_broadcasted - [0.0,0.0,0.0], axis=-1)
                
                # Volume of the Brillouin zone
            vbz = 1.0 / (np.prod(self.ngrid) * self.dir_vol)
            lc = self.lc
            Zc = 0.5 * self.rlat[2, 2]  # Half of the c lattice parameter

                # Compute differences between k-points
            vkpt = kpt1_broadcasted - kpt2_broadcasted
                
                # In-plane momentum transfer
            Qxy = np.sqrt(vkpt[..., 0]**2 + vkpt[..., 1]**2)
            Qz = np.sqrt(vkpt[..., 2]**2)   
                # Factor for the potential
            factor = 1.0  # This could also be 4.0 * pi, depending on the model
                
                # Compute the potential using vectorized operations
            v2dt2_array = np.where(
                modk < self.tolr,
                1.0,
                (vbz * self.alpha) * (factor / modk**2) *
                (1.0 - np.exp(-Qxy * Zc) * np.cos(Qz * Zc))
            )
            return v2dt2_array
        elif(self.ctype == 'v2dk'):
            #print('\n Kernel built from v2dk Coulomb potential. Remember to provide the cutoff length lc in Bohr\n')
            K_ex = self.cpot.v2dk(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.einsum('i,i->', self.eigvec[ik, :, ic].conj(), self.eigvec[ikminusq, :, iv])  \
                        *np.einsum('j,j->', self.eigvec[ikpminusq, :, ivp].conj(), self.eigvec[ikp, :, icp])          
            
        elif(self.ctype == 'vcoul'):
            #print('''\n Kernel built from screened Coulomb potential.\n
            #   Screening should be set via the instance of the Coulomb Potential class.\n
            #   ''')
            K_ex = self.cpot.vcoul(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.einsum('i,i->', self.eigvec[ik, :, ic].conj(), self.eigvec[ikminusq, :, iv])  \
                        *np.einsum('j,j->', self.eigvec[ikpminusq, :, ivp].conj(), self.eigvec[ikp, :, icp])          
        
        elif(self.ctype == 'v2dt'):
            #print('''\n Kernel built from v2dt Coulomb potential.\n
            #   ''')
            K_ex = self.cpot.v2dt(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0])\
                        *np.vdot(self.eigvec[ik,:, ic],self.eigvec[ikminusq,:, iv])*np.vdot(self.eigvec[ikpminusq,:, ivp],self.eigvec[ikp,: ,icp])
        
        elif(self.ctype == 'v2drk'):
            #print('''\n Kernel built from v2drk Coulomb potential.\n
            #   lc, ez, w and r0 should be set via the instance of the Coulomb potential class.\n
            #   ''')
            K_ex = self.cpot.v2drk(self.qmpgrid.car_kpoints[iq,:],[0.0,0.0,0.0] )\
                        *np.einsum('i,i->', self.eigvec[ik, :, ic].conj(), self.eigvec[ikminusq, :, iv])  \
                        *np.einsum('j,j->', self.eigvec[ikpminusq, :, ivp].conj(), self.eigvec[ikp, :, icp])          
        return K_ex
        
    def get_eps(self, hlm, emin, emax, estep, eta, method="Boltz", Tel=0.0, Tbos=300.0, sigma=0.1):
        '''
        Compute microscopic dielectric function 
        dipole_left/right = l/r_residuals.
        \eps_{\alpha\beta} = 1 + \sum_{kcv} dipole_left*dipole_right*(GR + GA)
        '''        
        w = np.arange(emin,emax,estep,dtype=np.float64)
        self.w = w.copy()
        F_kcv = np.zeros((self.dimbse,3,3), dtype=np.complex128)
        eps = np.zeros((len(w),3,3), dtype=np.complex128)
        pl0 = np.zeros((len(w),3,3), dtype=np.complex128)
        for i in range(eps.shape[0]):
            np.fill_diagonal(eps[i,:,:], 1.0)
        # First I have to compute the dipoles, then chi = 1 + FF*lorentzian

        h2peigvec_vck=self.h2peigvec_vck[self.q0index]
        h2peigv_vck = self.h2peigv_vck[self.q0index]
        h2peigvec = self.h2peigvec[self.q0index]
        h2peigv = self.h2peigv[self.q0index]

        #IP approximation, he doesn not haveh2peigvec_vck and then you call _get_dipoles()
        tb_dipoles = TB_dipoles(self.nc, self.nv, self.bse_nc, self.bse_nv, self.nk, self.eigv,self.eigvec, self.eta, hlm, self.T_table, self.BSE_table, h2peigvec, \
                                self.eigv_diff_ttp,self.eigvecc_t,self.eigvecv_t,mpgrid=self.model.mpgrid,cpot=self.cpot, h2peigv_vck= h2peigv_vck, h2peigvec_vck=h2peigvec_vck, method='real',ktype=self.ktype)
        # compute osc strength
        # self.dipoles_bse = tb_dipoles.dipoles_bse
        #self.dipoles = tb_dipoles.dipoles # ??? gargabe now
        if(self.ktype=='IP'):
            F_kcv = tb_dipoles.F_kcv
            self.F_kcv = F_kcv
            # self.dipoles_kcv = tb_dipoles.dipoles_kcv       #testing purposes
            self.dipoles_bse_kcv = tb_dipoles.dipoles_bse_kcv   #testing purposes
            # compute eps and pl
            #f_pl = TB_occupations(self.eigv,Tel = 0, Tbos=self.TBos, Eb=self.h2peigv[0])._get_fkn( method='Boltz')
            #pl = eps
            vbz = np.prod(self.cpot.ngrid)*self.electronsdb.lat_vol*bohr2ang**3
            for ies, es in enumerate(w):
                for t in range(0,self.dimbse):
                    ik = self.BSE_table_sort[0][t][0]
                    iv = self.BSE_table_sort[0][t][1]
                    ic = self.BSE_table_sort[0][t][2]
                    eps[ies,:,:] += 8*np.pi/(vbz)*F_kcv[ik,ic-self.nv,iv-self.offset_nv,:,:]*(np.real(h2peigv[t]-es))/(np.abs(es-h2peigv[t])**2+eta**2) \
                        + 1j*8*np.pi/(vbz)*F_kcv[ik,ic-self.nv,iv-self.offset_nv,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2)                     
                    #pl[ies,:,:] += f_pl * 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                    #    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2)            
            print('Excitonic Direct Ground state: ', np.min(h2peigv[:]), ' [eV]')
            #self.pl = pl
            #self.w = w
            #self.eps_0 = eps_0            
        else:
            F_kcv = tb_dipoles.F_kcv
            self.F_kcv = F_kcv
            # self.dipoles_kcv = tb_dipoles.dipoles_kcv       #testing purposes
            self.dipoles_bse_kcv = tb_dipoles.dipoles_bse_kcv   #testing purposes
            ediff = h2peigv[:, np.newaxis]-w[np.newaxis, :]
            ibz_factor = 1

                # ibz_factor = len(self.model.mpgrid.red_kpoints_full) / self.model.nk
                
            

            # compute eps and pl
            #f_pl = TB_occupations(self.eigv,Tel = 0, Tbos=self.TBos, Eb=self.h2peigv[0])._get_fkn( method='Boltz')
            #pl = eps
            # for ies, es in enumerate(w):
            #     for t in range(0,self.dimbse):
            #         ik = self.BSE_table_sort[0][t][0]
            #         iv = self.BSE_table_sort[0][t][1]
            #         ic = self.BSE_table_sort[0][t][2]
                    # eps[ies,:,:] += 8*np.pi/(self.electronsdb.lat_vol**bohr2ang**3*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                        # + 1j*8*np.pi/(self.electronsdb.lat_vol**bohr2ang**3*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
            #         #pl[ies,:,:] += f_pl * 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
            #         #    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
            weight_bse = np.zeros(self.ntransitions)+1
            if hasattr(self.model.mpgrid, 'red_kpoints_full'): # ibz case

                weight_bse = (self.model.mpgrid.kpoint_weights[self.BSE_table[:,0]])**(1/6)#*self.model.mpgrid.nkpoints

            vbz = np.prod(self.cpot.ngrid)*self.electronsdb.lat_vol*bohr2ang**3 #* ibz_factor**2
            piVk = 8*np.pi/(vbz)
            eps = piVk * np.einsum('txy,tw->wxy',F_kcv * weight_bse[:,None,None], (ediff)/(np.abs(ediff)**2+eta**2))
            eps += 1j*piVk * np.einsum('txy,tw->wxy',F_kcv * weight_bse[:,None,None], (eta)/(np.abs(ediff)**2+eta**2))
            
            f_pl = TB_occupations(self.h2peigv[0],Tel = Tel, Tbos=Tbos, Eb=self.h2peigv[0][0], sigma=sigma)._get_fkn(method=method)
            pl0 = piVk * np.einsum('txy,tw->wxy',f_pl[:,None,None]*F_kcv * weight_bse[:,None,None], (ediff)/(np.abs(ediff)**2+eta**2))
            pl0 += 1j*piVk * np.einsum('txy,tw->wxy',f_pl[:,None,None]*F_kcv * weight_bse[:,None,None], (eta)/(np.abs(ediff)**2+eta**2))
            # pl0 = eps + f_pl * piVk* F_kcv*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                        #  + 1j*piVk* F_kcv*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
            print('Excitonic Direct Ground state: ', np.min(h2peigv[:]), ' [eV]')
            #self.pl = pl
            # self.w = w
            # self.eps_0 = eps_0
        return w, eps, pl0
    
    def get_eps_yambo(self, hlm, emin, emax, estep, eta):
        '''
        Compute microscopic dielectric function 
        dipole_left/right = l/r_residuals.
        \eps_{\alpha\beta} = 1 + \sum_{kcv} dipole_left*dipole_right*(GR + GA)
        '''        
        w = np.arange(emin,emax,estep,dtype=np.float64)
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
        self.dipoles_bse = tb_dipoles.dipoles
        # self.dipoles = tb_dipoles.dipoles
        F_kcv = tb_dipoles.F_kcv
        self.F_kcv = F_kcv

        # compute eps and pl
        #f_pl = TB_occupations(self.eigv,Tel = 0, Tbos=self.TBos, Eb=self.h2peigv[0])._get_fkn( method='Boltz')
        #pl = eps
        for ies, es in enumerate(w):
            for t in range(0,self.dimbse):
                eps[ies,:,:] += 8*np.pi/(self.electronsdb.lat_vol*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                    + 1j*8*np.pi/(self.electronsdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
                #pl[ies,:,:] += f_pl * 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(h2peigv[t]-es)/(np.abs(es-h2peigv[t])**2+eta**2) \
                #    + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[t,:,:]*(eta)/(np.abs(es-h2peigv[t])**2+eta**2) 
        print('Excitonic Direct Ground state: ', np.min(h2peigv[:]), ' [eV]')
        #self.pl = pl
        # self.w = w
        # self.eps_0 = eps_0
        return w, eps

    # def _get_exc_overlap_ttp(self, t, tp , iq, ikq, ib):
    #     '''Calculate M_SSp(Q,B) = \sum_{ccpvvpk}A^{*SQ}_{cvk}A^{*SpQ}_{cpvpk+B/2}*<u_{ck}|u_{cpk+B/2}><u_{vp-Q-B/2}|u_{vk-Q}>'''     
    #     Mssp_ttp = 0
    #     for it in range(self.dimbse):
    #         for itp in range(self.dimbse):
    #             ik = self.BSE_table[it][0]
    #             iv = self.BSE_table[it][1] 
    #             ic = self.BSE_table[it][2] 
    #             ikp = self.BSE_table[itp][0]
    #             ivp = self.BSE_table[itp][1] 
    #             icp = self.BSE_table[itp][2] 
    #             iqpb = self.kmpgrid.qpb_grid_table[iq, ib][1]
    #             ikmq = self.kmpgrid.kmq_grid_table[ik,iq][1]
    #             ikpbover2 = self.kmpgrid.kpbover2_grid_table[ik, ib][1]
    #             ikmqmbover2 = self.kmpgrid.kmqmbover2_grid_table[ik, iq, ib][1]
    #             Mssp_ttp += np.conjugate(self.h2peigvec_vck[ikq,t,self.bse_nv-self.nv+iv,ic-self.nv,ik])*self.h2peigvec_vck[iqpb,tp,self.bse_nv-self.nv+ivp,icp-self.nv, ikpbover2]*\
    #                             np.vdot(self.eigvec[ik,:, ic], self.eigvec[ikpbover2,:, icp])*np.vdot(self.eigvec[ikmqmbover2,:,ivp], self.eigvec[ikmq,:,iv]) 
    #     return Mssp_ttp
    def _get_exc_overlap_ttp(self, t, tp, iq, ib):
        '''
        Calculate M_SSp(Q,B) = ∑_{cc'vv'k} A^{SQ*}_{cvk} A^{S'Q+B}_{c'v'k+B}
        × ⟨u_{ck}|u_{c'k+B}⟩ ⟨u_{v'k-Q-B}|u_{vk-Q}⟩
        '''
        indices = self.inverse_aux_t
        chunk_size = max(int(self.dimbse / 100.0), 1)

        ik_chunks = chunkify(self.BSE_table[indices, 0], chunk_size)
        iv_chunks = chunkify(self.BSE_table[indices, 1], chunk_size)
        ic_chunks = chunkify(self.BSE_table[indices, 2], chunk_size)

        Mssp_ttp = 0
        iqpb = self.qmpgrid.qpb_grid_table[iq, ib, 1]

        for ik, iv, ic in zip(ik_chunks, iv_chunks, ic_chunks):  # ∑_{cvk}
            ik = np.array(ik)[:, None]  # shape (N, 1)
            iv = np.array(iv)[:, None]
            ic = np.array(ic)[:, None]

            ikpb = self.kmpgrid.kpb_grid_table[ik, ib, 1]  # (N, 1)
            ikmq = self.kmpgrid.kmq_grid_table[ik, iq, 1]  # (N, 1)

            for ivp_chunk in iv_chunks:
                for icp_chunk in ic_chunks:
                    ivp = np.array(ivp_chunk)[None, :]  # shape (1, M)
                    icp = np.array(icp_chunk)[None, :]

                    # term1: A^{SQ*}_{cvk}
                    term1 = np.conjugate(self.h2peigvec_vck[iq, t, self.bse_nv - self.nv + iv, ic - self.nv, ik])  # shape (N, 1)

                    # term2: A^{S'Q+B}_{c'v'k+B}
                    term2 = self.h2peigvec_vck[iqpb, tp, self.bse_nv - self.nv + ivp, icp - self.nv, ikpb]  # shape (N, M)

                    # term3: ⟨u_c(k)|u_{c'}(k+B)⟩
                    u_c  = self.eigvec[ik, :, ic]     # shape (N, norb, 1)
                    u_cp = self.eigvec[ikpb, :, icp]  # shape (N, norb, M)
                    term3 = np.einsum('nij,nij->ni', np.conjugate(u_c), u_cp)  # shape (N, M)

                    # term4: ⟨u_{v'}(k-Q-B)|u_v(k-Q)⟩
                    u_v  = self.eigvec[ikmq, :, iv]   # shape (N, norb, 1)
                    u_vp = self.eigvec[ikmq, :, ivp]  # shape (N, norb, M)
                    term4 = np.einsum('nij,nij->ni', np.conjugate(u_vp), u_v)  # shape (N, M)

                    Mssp_ttp += np.sum(term1 * term2 * term3 * term4)  # scalar

        return Mssp_ttp

        # else:

        #     for ik, iv, ic in zip(ik_chunks, iv_chunks, ic_chunks): #\sum_{cvk}
        #         ik = np.array(ik)[:, np.newaxis]
        #         iv = np.array(iv)[:, np.newaxis]
        #         ic = np.array(ic)[:, np.newaxis]
        #         for ivp, icp in zip(iv_chunks,ic_chunks):   #\sum_{cpvp}
        #             #ivp = iv
        #             #icp = icp

        #             iqpb = self.qmpgrid.qpb_grid_table[iq, ib, 1] #points belong to qgrid
        #             #iqpb_ibz_ink = self.qgrid_toibzk[iqpb] # qpb point belonging in the IBZ going expressed in k grid
        #             ikmq = self.kminusq_table[ik, iq, 1] # points belong to k grids
        #             ikpbover2 = self.kmpgrid.kpbover2_grid_table[ik, ib, 1] # points belong to k grids
        #             ikmqmbover2 = self.kmpgrid.kmqmbover2_grid_table[ik, iq, ib, 1] # points belong to k grids
        #             #A^{*SQ}_{cvk}
        #             term1 = np.conjugate(self.h2peigvec_vck[iq, t, self.bse_nv - self.nv + iv, ic - self.nv, ik])
        #             #A^{SpQ+B}_{cpvpk+B/2}
        #             term2 = self.h2peigvec_vck[iqpb, tp, self.bse_nv - self.nv + ivp, icp - self.nv, ikpbover2]
        #             #<u_{ck}|u_{cpk+B/2}>_{uc}
        #             term3 = np.einsum('ijk,ijk->ij', np.conjugate(self.eigvec[ik, :, ic]), self.eigvec[ikpbover2, :, icp])
        #             #<u_{vpk-Q-B/2}|u_{vk-Q}>_{uc}
        #             term4 = np.einsum('ijk,ijk->ij', np.conjugate(self.eigvec[ikmqmbover2, :, ivp]), self.eigvec[ikmq, :, iv])
        #             Mssp_ttp += np.sum(term1 * term2 * term3 * term4)
        # return Mssp_ttp

    def convert_to_wannier90(self):
        '''
        This method translates the kpoints to wannier90, 
        and therefore the BSE_table
        '''
        y2w = self.kmpgrid.yambotowannier90_table
        # self.BSE_table[:,0] = y2w[self.BSE_table[:,0]]
        self.h2peigv = self.h2peigv[y2w]
        self.h2peigvec = self.h2peigvec[:,:,self.BSE_table[:,0]][y2w,:,:]
        self.h2peigv_vck = None
        self.h2peigvec_vck = self.h2peigvec_vck[:,:,:,:,y2w][y2w]
        print("*** Converted all the yambo grids to wannier90 grids. ***")




    def fix_and_align_phases(self):
        """
        Fixes both global phase (make first non-zero element real-positive)
        and aligns relative phase across Q using Q=0 as reference.
        
        A: np.ndarray with shape (nQ, nS, nv, nc, nk)
        Returns: phase-fixed A with the same shape
        """
        nQ, nS, nv, nc, nk = self.h2peigvec_vck.shape
        A_flat = self.h2peigvec_vck.reshape(nQ, nS, -1)

        # Step 1: Global phase fix — make first significant element real and positive
        ref_idx = np.argmax(np.abs(A_flat), axis=2)
        ref_vals = np.take_along_axis(A_flat, ref_idx[:, :, None], axis=2)[:, :, 0]
        global_phase = ref_vals / np.abs(ref_vals)
        A_flat = A_flat * np.conj(global_phase[:, :, None])

        # Step 2: Relative phase alignment — align all Q to Q=0
        A0 = A_flat[0]  # shape (nS, ntrans)
        for q in range(1, nQ):
            for s in range(nS):
                dot = np.vdot(A0[s], A_flat[q, s])  # complex scalar
                rel_phase = dot / np.abs(dot)
                A_flat[q, s] *= np.conj(rel_phase)

        print("*** Fixed global and relative phases across Q ***")
        return A_flat.reshape(nQ, nS, nv, nc, nk)
    
    

    def fast_exc_overlap_vectorized(self, t, tp, iq, ib):
        # Indices from BSE table
        ik  = self.BSE_table[:, 0]
        iv  = self.BSE_table[:, 1]
        ic  = self.BSE_table[:, 2]
        
        # Q and Q+B
        iqpb = self.qmpgrid.qpb_grid_table[iq, ib, 1]
        ikpb = self.kmpgrid.kpb_grid_table[ik, ib, 1]
        ikmq = self.kmpgrid.kmq_grid_table[ik, ib, 1]

        # Fetch relevant A vectors
        A1 = np.conj(self.h2peigvec_vck[iq, t, self.bse_nv - self.nv + iv, ic - self.nv, ik])
        A2 = self.h2peigvec_vck[iqpb, tp, self.bse_nv - self.nv + iv, ic - self.nv, ikpb]

        # Compute overlap terms efficiently
        # term3: ⟨u_c(k)|u_c(k+B)⟩
        u_c  = self.eigvec[ik, :, ic[:,None]]      # shape (n, norb)
        u_cp = self.eigvec[ikpb, :, ic[None,:]]    # shape (n, norb)
        term3 = np.einsum('ijk,ijk->ij', np.conj(u_c), u_cp)

        # term4: ⟨u_v(k-Q)|u_v(k-Q)⟩
        u_v  = self.eigvec[ikmq, :, iv[:,None]]    # shape (n, norb)
        u_vp = self.eigvec[ikmq, :, iv[None,:]]    # same index, because ivp = iv
        term4 = np.einsum('ijk,ijk->ij', np.conj(u_vp), u_v)

        return np.sum(A1[:,None] * A2[None,:] * term3 * term4)

    def get_exc_overlap(self, trange=[0], tprange=[0]):
        '''Calculate M_SSp(Q,B) = \sum_{ccpvvpk}A^{*SQ}_{cvk} A^{SpQ+B}_{cpvpk+B/2} \times <u_{ck}|u_{cpk+B/2}>_{uc}<u_{vpk-Q-B/2}|u_{vk-Q}>_{uc}'''
        '''\begin{aligned} M_{SS'}^{\alpha,\beta} = 
\sum_{cvc'v'k} A^{SQ\star}_{cvk}A^{S'Q+B}_{c'v'k+\alpha B} \bra{u_{ck}}\ket{u_{c'k+\alpha B}} \bra{u_{v'k-Q-\beta B}}\ket{u_{vk-Q}} \end{aligned}'''
        trange = np.array(trange)   # transition S range 
        tprange = np.array(tprange) # transition Sprime range
        self.h2peigvec_vck = self.fix_and_align_phases()
        it, itp, iq, ib = np.meshgrid(trange, tprange, np.arange(self.qmpgrid.nkpoints), np.arange(self.qmpgrid.nnkpts), indexing='ij')
        print(f"h2peigvec_vck count zeros: {self.h2peigvec_vck.size - np.count_nonzero(self.h2peigvec_vck)}")


        # vectorized_overlap_ttp_1 = np.vectorize(self.fast_exc_overlap_vectorized, signature='(),(),(),()->()')
        vectorized_overlap_ttp_2 = np.vectorize(self._get_exc_overlap_ttp, signature='(),(),(),()->()')

        # it itp: transition range
        # iq is the q index in the qgrid
        # kindices_table[iq] returns the index of the q-point in the k-grid
        # ib is the index of the neighbour in qgrid
        # Mssp_1 = vectorized_overlap_ttp_1(it, itp, iq, ib)/(self.ntransitions)**2  
        Mssp_2 = vectorized_overlap_ttp_2(it, itp, iq, ib)
        # self.check_A_norms(Mssp_1)
        # self.check_A_norms(Mssp_2/(self.ntransitions)**2)
        # self.check_hermitian(Mssp_1)
        # self.check_hermitian(Mssp_2/ (np.sqrt(self.ntransitions/2)))

        # print(print("Δ =", np.abs(Mssp_1 - Mssp_2)))        
        # self.Mssp = Mssp_2/(np.sqrt(self.ntransitions/2))      # under review
        self.Mssp = Mssp_2/(self.ntransitions*self.nv)      # under review


    def check_hermitian(self,M):
        dev = np.linalg.norm(M - M.conj().T)
        print(f"Hermitian deviation: {dev:.3e}")

    def check_A_norms(self,A):
        norms = np.linalg.norm(A, axis=(2,4))  # assuming shape: [nQ, nS, nv, nc=1, nk]
        print("Max deviation from 1:", np.max(np.abs(norms - 1)))



    def _get_amn_ttp(self, t, tp, iq):
        '''Calculate A_SSp(Q) = \sum_{ccpvvpk}A^{*SQ}_{cvk}<u_{ck}|u_{cpk}>'''
        #Extract indices from BSE_table
        if self.method == 'skip-diago':
            indices = self.inverse_aux_t
        else:
            indices = np.arange(0,self.dimbse,1)
        chunk_size=int(self.dimbse/100.0)
        if (chunk_size < 1): chunk_size=self.dimbse
        # Chunk the indices to manage memory usage
        ik_chunks = chunkify(self.BSE_table[indices, 0], chunk_size)
        iv_chunks = chunkify(self.BSE_table[indices, 1], chunk_size)
        ic_chunks = chunkify(self.BSE_table[indices, 2], chunk_size)

        Ammn_ttp = 0

        for ik, iv, ic in zip(ik_chunks, iv_chunks, ic_chunks):
            ik = np.array(ik)[:, np.newaxis]
            iv = np.array(iv)[:, np.newaxis]
            ic = np.array(ic)[:, np.newaxis]
            for ivp, icp in zip(iv_chunks,ic_chunks):
                #ivp = iv
                #icp = icp

                # iqpb = self.qmpgrid.qpb_grid_table[iq, ib, 1] #points belong to qgrid
                #iqpb_ibz_ink = self.qgrid_toibzk[iqpb] # qpb point belonging in the IBZ going expressed in k grid
                ikmq = self.kminusq_table[ik, iq, 1] # points belong to k grids
                ikplusq = self.kplusq_table[ik, iq, 1] # points belong to k grids
                term1 = np.conjugate(self.h2peigvec_vck[iq, t, self.bse_nv - self.nv + iv, ic - self.nv, ik])
                term2 = self.h2peigvec_vck[iq, tp, self.bse_nv - self.nv + ivp, icp - self.nv, ik]

                term3 = np.einsum('ijk,ijk->ij', np.conjugate(self.eigvec[ik, :, ic]), self.eigvec[ik, :, icp])
                term4 = np.einsum('ijk,ijk->ij', np.conjugate(self.eigvec[ikmq, :, ivp]), self.eigvec[ikmq, :, iv])
                Ammn_ttp += np.sum(term1 * term2  * term3 * term4 )
        #Ammn_ttp = np.sum(self.h2peigvec_vck[iq_ibz, t, :, :, :] * np.conjugate(self.h2peigvec_vck[iq_ibz, tp, :, :, :]))
        return Ammn_ttp

    def get_exc_amn(self, trange = [0], tprange = [0]):
        """Take the identiry projection of the excitons and store it in Amn"""
        Amn = np.zeros((len(trange), len(tprange),self.qmpgrid.nkpoints), dtype=np.complex128)
        trange = np.array(trange)
        tprange = np.array(tprange)
        for iq in range(self.qmpgrid.nkpoints):
            for m in range(min(len(trange), len(tprange))):
                Amn[m, m, iq] = 1.0 + 0.0j  # Identity projection

        self.Amn = Amn
        print(f"eigvec count: {np.count_nonzero(self.eigvec)}")
        print(f"h2peigvec_vck count: {np.count_nonzero(self.h2peigvec_vck)}")


    def write_exc_overlap(self, seedname='wannier90_exc', trange=[0], tprange=[0], alpha=1.0):
        self.alpha = alpha
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
        for iq in range(self.qmpgrid.nkpoints):
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
        with open(f'{seedname}_exc.mmn', 'w') as f_out:
            f_out.writelines(output_lines)

    def write_exc_eig(self, seedname='wannier90', trange = [0]):
        exc_eig = np.zeros((len(trange), self.qmpgrid.nkpoints), dtype=np.complex128)
        f_out = open(f'{seedname}_exc.eig', 'w')
        for iq in range(self.qmpgrid.nkpoints):
            #iq_ibz_ink = self.qgrid_toibzk[iq]
            for it,t in enumerate(trange):
                f_out.write(f'\t{it+1}\t{iq+1}\t{np.real(self.h2peigv[iq,it]):.13f}\n')
    
    def write_exc_win(self, seedname='wannier90', trange=[0]):
        input_file = f'{seedname}.win'
        output_file = f'{seedname}_exc.win'

        # Copy contents of seedname.win to seedname_exc.win
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                fout.write(line)

            fout.write('\nbegin nnkpts\n')
            for iq, q in enumerate(self.qmpgrid.k):
                for ib in range(self.qmpgrid.nnkpts):
                    iqpb = self.qmpgrid.qpb_grid_table[iq, ib][1]
                    Gx, Gy, Gz = self.qmpgrid.qpb_grid_table[iq, ib][2:5]
                    fout.write(f'\t{iq+1}\t{iqpb+1}\t{Gx}\t{Gy}\t{Gz}\n')
            fout.write('end nnkpts\n')


    def write_exc_nnkp(self, seedname='wannier90', trange = [0]):
        f_out = open(f'{seedname}_exc.nnkp', 'w')

        from datetime import datetime
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d at %H:%M:%S")
        f_out.write(f'Created on {date_time_string}\n\n')
        f_out.write('calc_only_A  :  F\n\n') # have to figure this one out
        
        f_out.write('begin real_lattice\n')
        for i, dim  in enumerate(self.kmpgrid.real_lattice):
            f_out.write(f'\t{dim[0]:.7f}\t{dim[1]:.7f}\t{dim[2]:.7f}\n')
        f_out.write('end real_lattice\n\n')

        f_out.write('begin recip_lattice\n')
        for i, dim in enumerate(self.kmpgrid.reciprocal_lattice):
            f_out.write(f'\t{dim[0]:.7f}\t{dim[1]:.7f}\t{dim[2]:.7f}\n')
        f_out.write('end recip_lattice\n\n')

        f_out.write(f'begin kpoints\n\t {len(self.qmpgrid.red_kpoints)}\n')
        for i, dat in enumerate(self.qmpgrid.red_kpoints):
            f_out.write(f'   {dat[0]:11.8f}\t{dat[1]:11.8f}\t{dat[2]:11.8f}\n')
        f_out.write(f'end kpoints\n\n')

        f_out.write(f'begin projections\n\t')
        f_out.write(f'\nend projections\n\n')
        
        f_out.write('begin nnkpts\n')
        f_out.write(f'\t{self.qmpgrid.nnkpts}\n')
        for iq, q in enumerate(self.qmpgrid.k):
            for ib in range(self.qmpgrid.nnkpts):
                iqpb = self.qmpgrid.qpb_grid_table[iq, ib][1]
                f_out.write(f'\t{iq+1}\t{iqpb+1}\t{self.qmpgrid.qpb_grid_table[iq,ib][2]}\t{self.qmpgrid.qpb_grid_table[iq,ib][3]}\t{self.qmpgrid.qpb_grid_table[iq,ib][4]}\n')
        f_out.write('end nnkpts')

    def write_exc_amn(self, seedname='wannier90', trange = [0], tprange = [0]):
        if (self.Amn is None):
            self.get_exc_amn(trange, tprange)

        f_out = open(f'{seedname}_exc.amn', 'w')

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
    def _get_aux_maps(self):
        yexc_atk = YamboExcitonDB.from_db_file(self.latdb, filename=f'{self.excitons_path}/ndb.BS_diago_Q1')
        aux_t = np.lexsort((yexc_atk.table[:, 2], yexc_atk.table[:, 1], yexc_atk.table[:, 0]))
        # Create an array to store the inverse mapping
        inverse_aux_t = np.empty_like(aux_t)
        # Populate the inverse mapping
        inverse_aux_t[aux_t] = np.arange(aux_t.size)        
        
        return aux_t, inverse_aux_t

def chunkify(lst, n):
    """Divide list `lst` into `n` chunks."""
    return [lst[i::n] for i in range(n)]    
