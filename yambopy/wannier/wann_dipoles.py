import numpy as np
from yambopy.wannier.wann_Gfuncs import GreensFunctions
from yambopy.wannier.wann_io import RMN
from yambopy.wannier.wann_utils import *


class TB_dipoles():
    '''dipoles = 1/(\DeltaE+ieta)*<c,k|P_\alpha|v,k>'''
    def __init__(self ,n_exc, nc, nv, bse_nc, bse_nv, nkpoints, eigv, eigvec, \
                 eta, hlm, T_table, BSE_table, h2peigvec,eigv_diff_ttp=None, eigvecc_t=None,eigvecv_t=None,\
                 mpgrid=None, cpot=None, \
                 h2peigv_vck = None, h2peigvec_vck = None,h2peigv=None, method = 'real',\
                 rmn = None,ktype='IP'):
        # hk, hlm are TBMODEL hamiltonians
        self.mpgrid = mpgrid
        self.cpot =cpot
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
        self.offset_nv = nv-bse_nv
        # self.eigvv = eigvv
        # self.eigvc = eigvc
        # self.eigvecv = eigvecv
        # self.eigvecc = eigvecc
        self.eta = eta
        self.hlm = hlm
        self.ktype=ktype
        self.eigv_diff_ttp = eigv_diff_ttp
        self.eigvecc_t = eigvecc_t
        self.eigvecv_t = eigvecv_t
        self.n_exc=n_exc
        if self.hlm[0][0][0][0] ==0:
            print(f"WARNING: hlm zero: {self.hlm[0][0][0][0]}")

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
        if (h2peigvec_vck is not None):
            self.h2peigvec_vck = h2peigvec_vck
            self.h2peigv_vck = h2peigv_vck
            self.h2peigvec = h2peigvec
            self.h2peigv = h2peigv
            # self.dipoles_bse = self._get_dipoles_bse(method)
            self.BSE_table = BSE_table
            if(self.ktype=='IP'):
                print('Running IP dipoles')
                #self._get_dipoles_IP(method=method)
            else:
                print('Running BSE dipoles')
                self._get_dipoles_bse(method=method)
        else:
            print('here')
            #self._get_dipoles(method=method)
        if(self.ktype=='IP'):
            print('Running IP oscillator strength')
            #self._get_osc_strength_IP(method)
        else:
            #self._get_osc_strength_IP(method)
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
            dipoles_kcv = np.zeros((self.nkpoints,self.nb,self.nb,3),dtype=np.complex128)
            for t in range(0,self.ntransitions):
                ik = self.T_table[t][0]
                iv = self.T_table[t][1]
                ic = self.T_table[t][2]
                w = self.eigv[ik,ic]
                E = self.eigv[ik,ic]
                GR = GreensFunctions(w=w, E=E, eta=self.eta).GR
                dipoles_kcv[ik, ic, iv,0] = GR*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                dipoles_kcv[ik, ic, iv,1] = GR*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                dipoles_kcv[ik, ic, iv,2] = GR*np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))

            self.dipoles_kcv = dipoles_kcv

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
            self.dipoles = dipoles#/(HA2EV**3)  

        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
           

    def _get_dipoles_bse(self, method):
        if method == 'real':
            import time
            print("Starting BSE dipole matrix formation\n")
            t0 = time.time()
            # transitions t = (#k * #c * #v)
            wc1 = self.eigv[self.BSE_table[:,0], self.BSE_table[:,2]]
            wv1 = self.eigv[self.BSE_table[:,0], self.BSE_table[:,1]]

            gr = GreensFunctions(w=wc1, E=wv1, eta=self.eta).GR      # (t,)
            ga = GreensFunctions(w=wc1, E=wv1, eta=self.eta).GA      # (t,)

            # Move to kernel subset
            BSE_TABLE = self.BSE_table.copy() - np.array([0, self.offset_nv, self.nv])

            # ---- only the first k excitons ----
            k = self.n_exc
            # A_{t,p} for the first k excitons (shape: nq x k x t) -> transpose to nq x t x k
            A_qkt = self.h2peigvec_vck[:, :k, BSE_TABLE[:,1], BSE_TABLE[:,2], BSE_TABLE[:,0]]
            nq = A_qkt.shape[0]

            # Transition-side contractions (unchanged); shapes: (t,3)
            dothlm     = np.einsum('tvca,tc->tva',  self.hlm[BSE_TABLE[:,0],:,:,:],
                                                self.eigvec[BSE_TABLE[:,0],:, self.BSE_table[:,1]],optimize=True)
            dothlm_conj= np.einsum('tvca,tc->tva',  self.hlm[BSE_TABLE[:,0],:,:,:],
                                                self.eigvec[BSE_TABLE[:,0],:, self.BSE_table[:,2]],optimize=True)

            vdot      = np.einsum('tv,tva->ta', np.conjugate(self.eigvec[BSE_TABLE[:,0],:, self.BSE_table[:,2]]), dothlm,optimize=True)        # (t,3)
            vdot_conj = np.einsum('tv,tva->ta', np.conjugate(self.eigvec[BSE_TABLE[:,0],:, self.BSE_table[:,1]]), dothlm_conj,optimize=True)   # (t,3)

            # Dipoles per transition t, exciton p, cart a, for each q (keep q explicit)
            # Broadcast gr/ga over the exciton axis k.
            dip      = gr[None, :,None] * np.einsum('qkt,ta->qkta', A_qkt, vdot, optimize=True)         # (nq, t, k, 3)
            dip_conj = ga[None, :,None] * np.einsum('qkt,ta->qkta', np.conjugate(A_qkt), vdot_conj, optimize=True)  # (nq, t, k, 3)

            # Reshape t -> (nkpoints, bse_nc, bse_nv)
            dipoles_bse_kcv      = dip.reshape(nq, k, self.nkpoints, self.bse_nc, self.bse_nv, 3)
            dipoles_bse_kcv_conj = dip_conj.reshape(nq, k, self.nkpoints, self.bse_nc, self.bse_nv, 3)

            # Store: now there’s an explicit exciton axis length k
            self.dipoles_bse_kcv = dipoles_bse_kcv
            self.dipoles_bse_kcv_conj = dipoles_bse_kcv_conj

            print(f"Done in {time.time()-t0:.3f}s (first {k} excitons)")


        print("BSE Dipoles matrix computed successfully.")
        print(f"Time for BSE Dipoles matrix formation: {time.time() - t0:.2f}")
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
            #self.dipoles = dipoles Perhaps if method is yambo we want to store dipoles differently                       

    def _get_dipoles_IP(self, method):
        if method == 'real':
            import time
            print("Starting BSE dipole matrix formation.\n")
            t0 = time.time()
            dipoles_bse_kcv = np.zeros((self.nkpoints, self.bse_nc,self.bse_nv,3),dtype=np.complex128)
            for tp in range(0,self.nbsetransitions): 
                ik = self.BSE_table[tp][0]
                iv = self.BSE_table[tp][1]
                ic = self.BSE_table[tp][2]
                w = self.eigv[ik,ic]
                E = self.eigv[ik,iv]
                GR = GreensFunctions(w=w, E=E, eta=self.eta).GR           
                GA = GreensFunctions(w=w, E=E, eta=self.eta).GA 
                #dipoles_bse_kck lives in the BSE kernel subset that's why we use the indices ic-self.nv and self.bse_nv-self.nv+iv
                dipoles_bse_kcv[ik, ic-self.nv, iv-self.offset_nv,0] = GR * \
                    np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,0],self.eigvec[ik,:,iv]))
                dipoles_bse_kcv[ik, ic-self.nv, iv-self.offset_nv,1] = GR * \
                    np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,1],self.eigvec[ik,:,iv]))
                dipoles_bse_kcv[ik, ic-self.nv, iv-self.offset_nv,2] = GR * \
                    np.vdot(self.eigvec[ik,:,ic],np.dot(self.hlm[ik,:,:,2],self.eigvec[ik,:,iv]))            
        self.dipoles_bse_kcv = dipoles_bse_kcv

        print("BSE Dipoles matrix computed successfully.")
        print(f"Time for BSE Dipoles matrix formation: {time.time() - t0:.2f}")
                
    def _get_osc_strength(self, method):
        """F_{αβ}^{n} = (Σ_{t} dipoles_{t,n,α}) * (Σ_{t} dipoles_{t,n,β})^*"""
        import time
        assert method == 'real', "only 'real' supported here"

        print('Computing oscillator strength')
        t0 = time.time()

        D  = self.dipoles_bse_kcv          # expected: (nq, k, nk, nc, nv, 3)
        Dc = self.dipoles_bse_kcv_conj     # same shape
        BSE_TABLE = self.BSE_table.copy() - np.array([0, self.offset_nv, self.nv])


        # Legacy shape without explicit exciton axis: (t, nk, nc, nv, 3)
        # Treat first dim as “exciton” and proceed similarly.
        J  = D[:,:,BSE_TABLE[:,0],BSE_TABLE[:,2],BSE_TABLE[:,1]].sum(axis=(2))             # (t, 3)
        Jc = Dc[:,:,BSE_TABLE[:,0],BSE_TABLE[:,2],BSE_TABLE[:,1]].sum(axis=(2))            # (t, 3)
        F  = np.einsum('qta,qtb->qtab', J, Jc, optimize=True)
        self.F_kcv = F


        print(f"Done in {time.time() - t0:.3f}s; F shape = {F.shape}")

        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        print(f"Oscillation strength computed succesfully in {time.time()-t0:.2f}s")

    def _get_osc_strength_IP(self,method):
        '''computes osc strength from dipoles
        '''
        print('Computing oscillator strength')
        import time
        t0 = time.time()
        dipoles_bse_kcv = self.dipoles_bse_kcv
        F_kcv = np.zeros((self.nkpoints,self.bse_nc,self.bse_nv,3, 3), dtype=np.complex128)   



        if (method == 'real'):
                tmp_F_left = np.zeros((self.nkpoints,self.bse_nc,self.bse_nv,3), dtype=np.complex128)
                tmp_F_right = np.zeros((self.nkpoints,self.bse_nc,self.bse_nv,3), dtype=np.complex128)
                for idip in range(0,self.nbsetransitions):
                    ik = self.BSE_table[idip][0]
                    iv = self.BSE_table[idip][1]
                    ic = self.BSE_table[idip][2]
                    factorLx = dipoles_bse_kcv[ik, ic-self.nv, iv-self.offset_nv, 0]
                    factorRx = factorLx.conj() 
                    factorLy = dipoles_bse_kcv[ik, ic-self.nv, iv-self.offset_nv, 1]
                    factorRy = factorLy.conj() 
                    factorLz = dipoles_bse_kcv[ik, ic-self.nv, iv-self.offset_nv, 2]
                    factorRz = factorLz.conj() 
                    tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,0] = factorLx
                    tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,1] = factorLy
                    tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,2] = factorLz
                    tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,0] = factorRx
                    tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,1] = factorRy
                    tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,2] = factorRz

                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,0,0] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,0]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,0]
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,0,1] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,0]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,1]    
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,0,2] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,0]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,2]    
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,1,0] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,1]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,0]
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,1,1] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,1]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,1]    
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,1,2] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,1]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,2]
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,2,0] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,2]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,0]
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,2,1] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,2]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,1]    
                    F_kcv[ik,ic-self.nv,iv-self.offset_nv,2,2] = tmp_F_left[ik,ic-self.nv,iv-self.offset_nv,2]*tmp_F_right[ik,ic-self.nv,iv-self.offset_nv,2]                                             

        if (method== 'v-gauge'):
            print('Warning! velocity gauge not implemented yet')
        if (method== 'r-gauge'):
            print('Warning! position gauge not implemented yet')
        if (method== 'covariant'):
            print('Warning! covariant approach not implemented yet')
        self.F_kcv = F_kcv        
        print(f"Oscillation strength computed succesfully in {time.time()-t0:.2f}s")