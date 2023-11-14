# Copyright (C) 2018 Henrique Pereira Coutada Miranda
# All rights reserved.
#
# This file is part of yambopy
#
# Author: Riccardo Reho 2023

import numpy as np
import multiprocessing
import tbmodels
from time import time
import typing as ty
from yambopy.wannier.wann_tb_mp import tb_Monkhorst_Pack
from yambopy.wannier.wann_Gfuncs import GreensFunctions
from yambopy.wannier.wann_utils import HA2EV, fermi_dirac, fermi_dirac_T
from yambopy.wannier.wann_dipoles import TB_dipoles
class TBMODEL(tbmodels.Model):
    '''
    Class that inherits from tbmodels.Model for TB-model Hamiltonians
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def set_mpgrid(cls,mpgrid):
        cls.mpgrid = mpgrid
    
    def solve_ham(self, k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]], convention: int = 2):
        # k in reduced coordinates
        self.H_k = self.hamilton(k, convention=convention)
        self.nk = int(self.H_k.shape[0]) # number of k-points
        self.nb = int(self.H_k.shape[1]) # number of bands
        # check hermiticity
        t0 = time()
        for k_index in range(self.H_k.shape[0]):
            if not np.allclose(self.H_k[k_index], self.H_k[k_index].T.conj(), atol=1e-9):
                raise ValueError(f"Warning! Hamiltonian matrix at k-point index {k_index} is not hermitian.")       
        # solve matrix
        # the eigenvalues only is already available within tbmodels, here I need the eigenvectors
        # find eigenvalues and eigenvectors
        self.eigv= np.zeros((self.nk,self.nb),dtype=np.complex128)
        self.eigvec = np.zeros((self.nk,self.nb,self.nb),dtype=np.complex128)
        for ik in range(self.nk):
            (self.eigv[ik], self.eigvec[ik]) = np.linalg.eigh(self.H_k[ik])
            (self.eigv[ik],self.eigvec[ik]) = self._sort_eig(self.eigv[ik],self.eigvec[ik])
        # transpose to have eigvec[:,i] associated with eval[i]
        # one could transpose it to have the opposite, for now commented out bcs I don't like it
        #self.eig = self.eig.T
        # sort eigenvectors 
        # To-do should I check for the spin case?
        t1 = time()
        print(f'Diagonalization took {t1-t0:.3f} sec')

    def solve_ham_from_hr(self, latdb, hr, fermie):
        # k in reduced coordinates
        self.latdb = latdb
        nkpt = self.mpgrid.nkpoints
        H_k = np.zeros((nkpt, hr.num_wann, hr.num_wann), dtype=np.complex128)
        for i in range(0,nkpt):
            H_k[i] = self._get_h_k(self.mpgrid.car_kpoints[i], latdb.lat, hr, fermie, from_hr=True)
        
        self.H_k = H_k

        self.nk = int(self.H_k.shape[0]) # number of k-points
        self.nb = int(self.H_k.shape[1]) # number of bands
        # check hermiticity
        t0 = time()
        for k_index in range(self.H_k.shape[0]):
            if not np.allclose(self.H_k[k_index], self.H_k[k_index].T.conj(), atol=1e-9):
                raise ValueError(f"Warning! Hamiltonian matrix at k-point index {k_index} is not hermitian.")       
        # solve matrix
        # the eigenvalues only is already available within tbmodels, here I need the eigenvectors
        # find eigenvalues and eigenvectors
        self.eigv= np.zeros((self.nk,self.nb),dtype=np.complex128)
        self.eigvec = np.zeros((self.nk,self.nb,self.nb),dtype=np.complex128)
        for ik in range(self.nk):
            (self.eigv[ik], self.eigvec[ik]) = np.linalg.eigh(self.H_k[ik])
            (self.eigv[ik],self.eigvec[ik]) = self._sort_eig(self.eigv[ik],self.eigvec[ik])
        
        self._get_occupations(self.nk, self.nb, self.eigv, fermie)
        self._reshape_eigvec(self.nk, self.nb, self.f_kn, self.eigv, self.eigvec)
        self._get_T_table()
        

        # transpose to have eig[:,i] associated with eval[i]
        # one could transpose it to have the opposite, for now commented out bcs I don't like it
        #self.eig = self.eig.T
        # sort eigenvectors 
        # To-do should I check for the spin case?

        t1 = time()
        print(f'Diagonalization took {t1-t0:.3f} sec')

    @classmethod
    def get_pos_from_ham(self, lat, hr, from_hr = True):
        'get positions from Hamiltonian, first the indices and then cartesian coordinates of hoppings'
        # get tuple of irpos
        if (not from_hr):
            self.irpos = np.array(list(self.hop.keys())) 
            self.nrpos = len(self.irpos)
            self.lat = lat
            # pos[i,3] position of i unit cells. Store only the upper triangular matrix
            self.pos = np.dot(self.lat, self.irpos.T).T
            return self.pos
        else:
            self.lat = lat
            self.pos = np.dot(self.lat, hr.hop.T).T
            self.nrpos = hr.nrpts
            return self.pos

    def get_hlm (self ,lat, hr, from_hr=True):
        ''' computes light mater interaction hamiltonian for grid of points
        k is one k-point in cartesian coordinates
        hrx = P_\alpha = dH(k)\dk_\alpha = \sum_{R=1}^N e^{ikR}iR_\alpha H_{R}
        here we get iR_\alpha*H_{R} as h_x,hy,hz
        ''' 
  
        hlm = np.zeros((self.mpgrid.nkpoints,hr.num_wann, hr.num_wann,3), dtype=np.complex128)       
  
        for i,k in enumerate(self.mpgrid.car_kpoints):
            hlm[i] = self._get_hlm_k(k,lat,hr,from_hr=True)          

        self.hlm = hlm

    def _get_hlm_k(self, k, lat, hr, from_hr=True):
        ''' computes light mater interaction hamiltonian at k
        k is one k-point in cartesian coordinates
        hrx = P_\alpha = dH(k)\dk_\alpha = \sum_{R=1}^N e^{ikR}iR_\alpha H_{R}
        here we get iR_\alpha*H_{R} as h_x,hy,hz
        '''
        #ws_deg is needed for fourier factor
        #to do, make it work also for hr from tbmodels
        # I started but I do not have ffactor from tbmodels.
        if (not from_hr):
            pos = self.get_pos_from_ham(lat)
        else:
            pos = self.get_pos_from_ham(lat, hr, from_hr=True)
            irpos = hr.hop

        hlm_k = np.zeros((self.nb, self.nb, 3), dtype=np.complex128)
        kvecs = np.zeros(self.nrpos, dtype=np.complex128)
        kvecsx = np.zeros(self.nrpos, dtype=np.complex128)
        kvecsy = np.zeros(self.nrpos, dtype=np.complex128)
        kvecsz = np.zeros(self.nrpos, dtype=np.complex128)
        #pos has shape (nrpts,3)
        kvecs[:] = 1j*np.dot(pos, k)
        kvecsx = 1j*pos[:,0]
        kvecsy = 1j*pos[:,1]
        kvecsz = 1j*pos[:,2]


        # len(pos) = nrpts
        for i in range(0,self.nrpos):
            if (np.array_equal(irpos[i],[0.0,0.0,0.0])):
                hr_mn_p = hr.HR_mn[i,:,:]
                np.fill_diagonal(hr_mn_p,complex(0.0))
                hlm_k[:,:,0] = hlm_k[:,:,0] +kvecsx[i]*np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,1] = hlm_k[:,:,1] +kvecsy[i]*np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,2] = hlm_k[:,:,2] +kvecsz[i]*np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
            else:
                hr_mn_p = hr.HR_mn[i,:,:]
                hlm_k[:,:,0] = hlm_k[:,:,0] +kvecsx[i]*np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,1] = hlm_k[:,:,1] +kvecsy[i]*np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,2] = hlm_k[:,:,2] +kvecsz[i]*np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
        
        return hlm_k

    @classmethod
    def _get_h_k(cls, k, lat, hr,fermie, from_hr=True):
        ''' computes hamiltonian at k from real space hamiltonian
        k is one k-point in cartesian coordinates
        '''
        #ws_deg is needed for fourier factor
        #to do, make it work also for hr from tbmodels
        # I started but I do not have ffactor from tbmodels.
        if (not from_hr):
            cls.get_pos_from_ham(lat)
        else:
            cls.get_pos_from_ham(lat=lat,hr=hr, from_hr=True)
            irpos = hr.hop
        
        pos = cls.pos

        hk = np.zeros((hr.num_wann, hr.num_wann), dtype=np.complex128)
        kvecs = np.zeros(cls.nrpos, dtype=np.complex128)
        #pos has shape (nrpts,3)
        kvecs[:] = 1j*np.dot(pos, k)


        # len(pos) = nrpts
        for i in range(0,cls.nrpos):
            hr_mn_p = hr.HR_mn[i,:,:]
            hk[:,:] = hk[:,:] +np.exp(kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
        
        hk = hk #+ fermie
        return hk

    @classmethod
    def pos_operator_matrix(cls, eigvec, dir, cartesian = True):
        ''' Computes the position operator along a direction dir at a k-point
            position operator is returned cartesian coordinates by default
            X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
            r^{\alpha} \vert u_{n {\bf k}} \rangle            
        ''' 
        pass
    


    @classmethod
    def _sort_eig(cls,eigv,eigvec=None):
        "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
        # first take only real parts of the eigenvalues
        eigv=np.array(eigv.real,dtype=float)
        # sort energies
        args=eigv.argsort()
        eigv=eigv[args]
        if not (eigvec is None):
            eigvec=eigvec[args]
            return (eigv,eigvec)
        return eigv
    
    @classmethod
    def _reshape_eigvec(cls, nk, nb, f_kn, eigv, eigvec):
        nv = np.count_nonzero(f_kn[0])
        nc = nb -nv
        print(nv,nc)
        eigvecv = np.zeros((nk, nb, nv),dtype=np.complex128)
        eigvecc = np.zeros((nk, nb, nc), dtype=np.complex128)
        eigvv = np.zeros((nk,nv),dtype=np.complex128)
        eigvc = np.zeros((nk,nc), dtype = np.complex128)
        for n in range(0,f_kn.shape[1]):
            for k in range(0, f_kn.shape[0]):
                if(f_kn[k,n] > 0.5):
                    eigvv[k,n] = eigv[k,n]
                    eigvecv[k,:,n] = eigvec[k,:,n]
                else:
                    eigvc[k,n-nc] = eigv[k,n]
                    eigvecc[k,:,n-nc] = eigvec[k,:,n]
        cls.eigvecv = eigvecv
        cls.eigvecc = eigvecc
        cls.eigvv = eigvv
        cls.eigvc = eigvc
        cls.nc = nc
        cls.nv = nv

    def _get_T_table(self):
        ntransitions = self.nk*self.nc*self.nv
        T_table = np.zeros((ntransitions, 3),dtype=int)
        t_index = 0
        for ik in range(0,self.nk):
            for iv in range(0,self.nv):
                for ic in range(0,self.nc):
                        T_table[t_index] = [ik, iv, self.nv+ic]
                        t_index += 1
        self.T_table = T_table

    def get_eps_0(self, hlm, emin, emax, estep, eta, eigvecv, eigvecc):
        w = np.arange(emin,emax,estep,dtype=np.float32)
        chi = np.zeros(len(w), dtype=np.complex128)
        F_kcv = np.zeros((3,3), dtype=np.complex128)
        eps_0 = np.zeros((len(w),3,3), dtype=np.complex128)
        E_0 = np.min(np.min(self.eigvc)-np.max(self.eigvv))
        for i in range(eps_0.shape[0]):
            np.fill_diagonal(eps_0[i,:,:], 1)
        # First I have to compute the dipoles, then chi = 1 + FF*lorentzian
        dipoles = TB_dipoles(self.nk*self.nv*self.nc, self.nc, self.nv, self.nk,self.eigv,self.eigvec, eta, hlm, self.T_table).dipoles
        for ik in range(0,self.nk):
            for iv in range(self.nv):
                for ic in range(self.nv,self.nc+self.nv):
                    #E = self.eigv[k,c] - self.eigv[k,v]
                    #GFs = GreensFunctions(w,E,eta)
                    #GR = GFs.G_retarded()
                    #GA = GFs.G_advanced()
                    factorRx = dipoles[ik,ic,iv,0]#1/(E +1j*eta)*np.dot(eigvecc[k,:,c].conj().T,np.dot(hlm[k,:,:,0],eigvecv[k,:,v]))
                    factorLx = factorRx.conj() 
                    factorRy = dipoles[ik,ic,iv,1]#1/(E +1j*eta)*np.dot(eigvecc[k,:,c].conj().T,np.dot(hlm[k,:,:,1],eigvecv[k,:,v]))
                    factorLy = factorRy.conj() 
                    factorRz = dipoles[ik,ic,iv,2]#1/(E +1j*eta)*np.dot(eigvecc[k,:,c].conj().T,np.dot(hlm[k,:,:,2],eigvecv[k,:,v]))
                    factorLz = factorRz.conj() 
                    F_kcv[0,0] = F_kcv[0,0] + factorRx*factorLx
                    F_kcv[0,1] = F_kcv[0,1] + factorRx*factorLy
                    F_kcv[0,2] = F_kcv[0,2] + factorRx*factorLz
                    F_kcv[1,0] = F_kcv[1,0] + factorRy*factorLx
                    F_kcv[1,1] = F_kcv[1,1] + factorRy*factorLy
                    F_kcv[1,2] = F_kcv[1,2] + factorRy*factorLz                    
                    F_kcv[2,0] = F_kcv[2,0] + factorRz*factorLx
                    F_kcv[2,1] = F_kcv[2,1] + factorRz*factorLy
                    F_kcv[2,2] = F_kcv[2,2] + factorRz*factorLz  
        for ies, es in enumerate(w):
            eps_0[ies,:,:] += 8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[:,:]*(E_0-es)/(np.abs(es-E_0)**2+eta**2) \
                           + 1j*8*np.pi/(self.latdb.lat_vol*self.nk)*F_kcv[:,:]*(eta)/(np.abs(es-E_0)**2+eta**2) 
        print('Direct Ground state: ',E_0, ' [eV]')
        # self.w = w
        # self.eps_0 = eps_0
        return w, eps_0

    @classmethod
    def _get_occupations(cls, nk, nb, eigv, fermie):
        occupations = np.zeros((nk, nb))
        occupations = fermi_dirac(eigv,fermie)
        cls.f_kn = np.real(occupations)
    
    def get_eps(self):
        '''
        Compute microscopic dielectric function 
        dipole_left/right = l/r_residuals.
        \eps_{\alpha\beta} = 1 + \sum_{kcv} dipole_left*dipole_right*(GR + GA)
        '''

