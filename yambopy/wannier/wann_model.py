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
from yambopy.wannier.wann_asegrid import ase_Monkhorst_Pack
from yambopy.wannier.wann_Gfuncs import GreensFunctions
from yambopy.wannier.wann_utils import HA2EV, fermi_dirac, fermi_dirac_T, sort_eig
from yambopy.wannier.wann_dipoles import TB_dipoles
import matplotlib.pyplot as plt
import scipy
class TBMODEL(tbmodels.Model):
    """
    Class that inherits from tbmodels.Model for TB-model Hamiltonians.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Mmn = None
    
    def supercell(  # pylint: disable=too-many-locals
        self, size: ty.Sequence[int]
    ) -> tbmodels.Model:
        """Generate a model for a supercell of the current unit cell.

        Parameters
        ----------
        size : 
            The size of the supercell, given as integer multiples of the
            current lattice vectors
        """
        import itertools
        import collections as co

        HoppingType = ty.Dict[ty.Tuple[int, ...], ty.Any]

        size_array = np.array(size).astype(dtype=int, casting="safe")
        if size_array.shape != (self.dim,):
            raise ValueError(
                "The given 'size' has incorrect shape {}, should be {}.".format(
                    size_array.shape, (self.dim,)
                )
            )
        volume_multiplier = np.prod(size_array)
        new_occ = None if self.occ is None else volume_multiplier * self.occ
        if self.uc is None:
            new_uc = None
        else:
            new_uc = (self.uc.T * size_array).T

        # the new positions, normalized to the supercell
        new_pos: ty.List[np.ndarray] = []
        reduced_pos = np.array([p / size_array for p in self.pos])
        uc_offsets = list(
            np.array(offset)
            for offset in itertools.product(*[range(n) for n in size_array])
        )
        for current_uc_offset in uc_offsets:
            new_pos.extend(reduced_pos + (current_uc_offset / size_array))

        new_size = self.size * volume_multiplier
        new_hop: HoppingType = co.defaultdict(
            lambda: np.zeros((new_size, new_size), dtype=complex)
        )

        # Can be used to get the orbital offset of a given unit cell
        # by taking the inner product with the unit cell position.
        uc_idx_multiplier = (
            np.array([np.prod(size[i:], dtype=int) for i in range(1, len(size) + 1)])
            * self.size
        )

        for uc1_idx, uc1_pos in enumerate(uc_offsets):
            uc1_idx_offset = uc1_idx * self.size

            for R, hop_mat in self.hop.items():
                hop_mat = self._array_cast(hop_mat)

                # position of the uc of orbital 2, not mapped inside supercell
                full_uc2_pos = uc1_pos + R
                # mapped into the supercell
                uc2_pos = full_uc2_pos % size_array
                uc2_idx_offset = np.inner(uc_idx_multiplier, uc2_pos)

                # R in terms of supercells
                new_R = np.array(np.floor(full_uc2_pos / size_array), dtype=int)

                new_hop[tuple(new_R)][
                    uc1_idx_offset : uc1_idx_offset + self.size,
                    uc2_idx_offset : uc2_idx_offset + self.size,
                ] += hop_mat

        return TBMODEL(
            **co.ChainMap(
                dict(
                    hop=new_hop,
                    occ=new_occ,
                    uc=new_uc,
                    size=new_size,
                    pos=new_pos,
                    contains_cc=False,
                ),
                self._input_kwargs,
            )
        )


    @classmethod
    def set_mpgrid(cls,mpgrid):
        """
        Set the Monkhorst-Pack grid.

        Parameters:
            mpgrid: The Monkhorst-Pack grid.
        """        
        cls.mpgrid = mpgrid
    
    def solve_ham(self, k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]], convention: int = 2):
        """
        Solve the Hamiltonian.

        Parameters:
            k: The k-points in reduced coordinates.
            convention: The convention to use.
        """        
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
            (self.eigv[ik], self.eigvec[ik]) = scipy.linalg.eigh(self.H_k[ik])
            #(self.eigv[ik],self.eigvec[ik]) = sort_eig(self.eigv[ik],self.eigvec[ik])
        # transpose to have eigvec[:,i] associated with eval[i]
        # one could transpose it to have the opposite, for now commented out bcs I don't like it
        #self.eig = self.eig.T
        # sort eigenvectors 
        # To-do should I check for the spin case?
        t1 = time()
        print(f'Diagonalization took {t1-t0:.3f} sec')

    def solve_ham_from_hr(self, latdb, hr, fermie):
        # k in reduced coordinates
        """
        Solve the Hamiltonian matrix with the k-points defined in the input 
        Monkhorst-Pack grid.

        Parameters:
            latdb: The lattice database.
            hr: The Hamiltonian matrix in Wannier90 format.
            fermie: The Fermi energy.

        Returns:
            The eigenvalues and eigenvectors of the Hamiltonian matrix at the k-points.
        """
        self.latdb = latdb
        nkpt = self.mpgrid.nkpoints
        H_k = np.zeros((nkpt, hr.num_wann, hr.num_wann), dtype=np.complex128)
        for i in range(0,nkpt):
            H_k[i] = self._get_h_k(self.mpgrid.k[i], latdb.lat, hr, fermie, from_hr=True)
        
        self.hr = hr
        self.H_k = H_k
        self.fermie = fermie
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
            (self.eigv[ik], self.eigvec[ik]) = scipy.linalg.eigh(self.H_k[ik])
            #(self.eigv[ik],self.eigvec[ik]) = sort_eig(self.eigv[ik],self.eigvec[ik])
        self._get_occupations(self.nk, self.nb, self.eigv, fermie)
        self._reshape_eigvec(self.nk, self.nb, self.f_kn, self.eigv, self.eigvec)
        self._get_T_table()
        self._build_Umn()
        #self.r_mnk = self.pos_operator_matrix(self.eigvec, )
        

        # transpose to have eig[:,i] associated with eval[i]
        # one could transpose it to have the opposite, for now commented out bcs I don't like it
        #self.eig = self.eig.T
        # sort eigenvectors 
        # To-do should I check for the spin case?

        t1 = time()
        print(f'Diagonalization took {t1-t0:.3f} s')

    def get_pos_from_ham(self, lat, hr, from_hr = True):
        'get positions from Hamiltonian, first the indices and then reduced coordinates of hoppings'
        # get tuple of irpos
        if (not from_hr):
            self.irpos = np.array(list(self.hop.keys())) 
            self.nrpos = len(self.irpos)
            self.lat = lat
            # pos[i,3] position of i unit cells. Store only the upper triangular matrix
            self.pos = self.irpos
            return self.pos
        else:
            self.lat = lat
            pos = hr.hop
            self.nrpos = hr.nrpts
            self.pos = pos
            return pos

    def get_hlm(self ,lat, hr, from_hr=True):
        ''' computes light mater interaction hamiltonian for grid of points
        k is one k-point in reduced coordinates
        hrx = P_\alpha = dH(k)\dk_\alpha = \sum_{R=1}^N e^{ikR}iR_\alpha H_{R}
        here we get iR_\alpha*H_{R} as h_x,hy,hz
        '''
  
        hlm = np.zeros((self.mpgrid.nkpoints,hr.num_wann, hr.num_wann,3), dtype=np.complex128)       
  
        for i,k in enumerate(self.mpgrid.red_kpoints):
            hlm[i] = self._get_hlm_k(k,lat,hr,from_hr=True)          

        self.hlm = hlm

    def _get_hlm_k(self, k, lat, hr, from_hr=True):
        ''' computes light mater interaction hamiltonian at k
        k is one k-point in reduced coordinates
        hrx = P_\alpha = dH(k)\dk_\alpha = \sum_{R=1}^N e^{ikR}iR_\alpha H_{R}
        here we get iR_\alpha*H_{R} as h_x,hy,hz
        '''
        #ws_deg is needed for fourier factor
        #to do, make it work also for hr from tbmodels
        # I started but I do not have ffactor from tbmodels.
        if (not from_hr):
            pos = self.get_pos_from_ham(lat, hr, from_hr)
        else:
            pos = self.get_pos_from_ham(lat, hr, from_hr=True)
            irpos = hr.hop
            self.pos = pos
            self.irpos = irpos

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
                hr_mn_p = np.copy(hr.HR_mn[i,:,:])
                np.fill_diagonal(hr_mn_p,np.complex128(0.0))
                hlm_k[:,:,0] += kvecsx[i]*np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,1] += kvecsy[i]*np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,2] += kvecsz[i]*np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
            else:
                hr_mn_p = hr.HR_mn[i,:,:]
                hlm_k[:,:,0] += kvecsx[i]*np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,1] += kvecsy[i]*np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
                hlm_k[:,:,2] += kvecsz[i]*np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
        
        return hlm_k


    def _get_h_k(self, k, lat, hr, fermie, from_hr=True):
        '''
        This mehod computes the Hamiltonian matrix hk at a given k-point k in reduced coordinates,
        using the real-space Hamiltonian hr. It does this by:

        1) Getting the positions pos from the Hamiltonian hr (either from hr directly or by calling get_pos_from_ham).
        2)Computing the Fourier factor kvecs by taking the dot product of pos and k.
        3)Iterating over the positions, adding the contribution of each position to the Hamiltonian matrix hk using the Fourier factor and the real-space Hamiltonian hr.
        
        The result is the Hamiltonian matrix hk at the given k-point k. The Fermi energy fermie is
        not currently added to the result, as indicated by the commented-out line.        
        '''
        #ws_deg is needed for fourier factor
        #to do, make it work also for hr from tbmodels
        # I started but I do not have ffactor from tbmodels.
        if (not from_hr):
            pos = self.get_pos_from_ham(lat, hr, from_hr)
        else:
            pos = self.get_pos_from_ham(lat=lat,hr=hr, from_hr=True)
            irpos = hr.hop
            self.irpos = irpos
        
        hk = np.zeros((hr.num_wann, hr.num_wann), dtype=np.complex128)
        kvecs = np.zeros(self.nrpos, dtype=np.complex128)
        #pos has shape (nrpts,3)
        kvecs[:] = 1j*np.dot(pos, k)

        # len(pos) = nrpts
        for i in range(0,self.nrpos):
            hr_mn_p = hr.HR_mn[i,:,:]
            hk[:,:] = hk[:,:] + np.exp(2*np.pi*kvecs[i])*(hr_mn_p)*(1.0/hr.ws_deg[i])
        
        hk = hk #+ fermie
        return hk

    def _get_h_R(self, lat, hr, fermie, from_hr=True):
        #ws_deg is needed for fourier factor
        #to do, make it work also for hr from tbmodels
        # I started but I do not have ffactor from tbmodels.
        '''
        This method returns the real-space Hamiltonian matrix hr_mn_p at R, by iterating over the positions and adding the contribution 
        of each position to the Hamiltonian matrix using the Fourier factor and the real-space Hamiltonian hr.
        
        Parameters:
        k (numpy array): k-point in reduced coordinates
        lat (3x3 array): lattice vectors
        hr (tbmodels class): tight-binding Hamiltonian
        fermie (float): Fermi energy
        from_hr (bool): if True, read from tbmodels, if False from Wannier90 output
        
        Returns:
        hr_mn_p (numpy array): real-space Hamiltonian matrix at R
        '''
        if (not from_hr):
            pos = self.get_pos_from_ham(lat, hr, from_hr)
        else:
            pos = self.get_pos_from_ham(lat=lat,hr=hr, from_hr=True)
            irpos = hr.hop
            self.irpos = irpos        
        # len(pos) = nrpts
        hr_mn_p = hr.HR_mn[:,:,:]
        return hr_mn_p    
    
    def decay_R(self, lat, hr ,fermie, from_hr = True):
        """
        Calculate the decay of the Hamiltonian matrix elements with respect to the distance R.

        This method computes the maximum absolute value of the Hamiltonian matrix elements 
        for each real-space lattice vector R and plots them against the distance |R|.

        Parameters
        ----------
        lat : array_like
            The lattice vectors.
        hr : object
            The Hamiltonian object containing Hamiltonian matrix elements and related data.
        fermie : float
            The Fermi energy.
        from_hr : bool, optional
            If True, compute using the real-space Hamiltonian (default: True).

        Returns
        -------
        R_dist : ndarray
            The distances of each lattice vector R.
        max_hr_p : ndarray
            The maximum absolute values of the Hamiltonian matrix elements at each R.
        """
        hr_mn_p = self._get_h_R(lat, hr, fermie, from_hr)
        #calculate distances
        max_hr_p = np.zeros(hr.nrpts, dtype= np.float64)
        R_dist = np.zeros(hr.nrpts, dtype= np.float64)
        for i in range(self.nrpos):
            R_dist[i] = np.linalg.norm(self.pos[i])
            max_hr_p[i] = np.max(np.abs(hr_mn_p[i]))
        fig, ax = plt.subplots()
        ax.set_xlabel('R [Bohr]')
        ax.set_ylabel(r'max $H_{MN}(R)$')
        #sorting and unique
        tmpidx = np.argsort(R_dist)
        R_dist = R_dist[tmpidx]
        max_hr_p = max_hr_p[tmpidx]
        
        ax.plot(R_dist, max_hr_p)

        return R_dist, max_hr_p

    def get_eigenval(self, k, from_hr=True):
        """
        Returns the eigenvalues at a given k point, or list of k-points.

        Parameters
        ----------
        k :
            The k-point at which the Hamiltonian is evaluated. If a list
            of k-points is given, a corresponding list of eigenvalue
            arrays is returned.
        """
        H_k = np.zeros((k.shape[0], self.nb, self.nb), dtype=np.complex128)
        for i in range(0, k.shape[0]):
            H_k[i] = self._get_h_k(k[i], self.latdb.lat, self.hr, self.fermie, from_hr)
        return np.array([scipy.linalg.eigvalsh(ham) for ham in H_k])

    def get_eigenval_and_vec(self, k, from_hr=True):
        """
        Returns the eigenvalues at a given k point, or list of k-points.

        Parameters
        ----------
        k : ndarray
            The k-point(s) at which the Hamiltonian is evaluated. If a list
            of k-points is given, a corresponding list of eigenvalue and 
            eigenvector arrays is returned.

        from_hr : bool, optional
            Whether to evaluate the Hamiltonian from the `hr` (default: True).
        Returns
        -------
        list of tuples
            A list where each element is a tuple of (eigenvalues, eigenvectors)
            for the corresponding k-point. The eigenvalues are sorted in ascending 
            order, and the eigenvectors are in column form.
        """
        # Initialize the Hamiltonian for all k-points
    # Ensure input is a NumPy array
        if not isinstance(k, np.ndarray):
            raise TypeError("Input `k` must be a NumPy array.")

        # Check dimensionality of `k`
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError("Input `k` must have shape (n_kpoints, 3).")

        # Initialize the Hamiltonian for all k-points
        H_k = np.zeros((k.shape[0], self.nb, self.nb), dtype=np.complex128)
        for i in range(k.shape[0]):
            H_k[i] = self._get_h_k(k[i], self.latdb.lat, self.hr, self.fermie, from_hr)

        # Compute eigenvalues and eigenvectors for each k-point
        eigenvalues = np.zeros((k.shape[0], self.nb), dtype=np.float64)
        eigenvectors = np.zeros((k.shape[0], self.nb, self.nb), dtype=np.complex128)

        for i, ham in enumerate(H_k):
            eigvals, eigvecs = scipy.linalg.eigh(ham)
            eigenvalues[i] = eigvals
            eigenvectors[i] = eigvecs

        return eigenvalues, eigenvectors
        
    def pos_operator_matrix(self, eigvec, cartesian = True):
        ''' Computes the position operator along a direction dir at a k-point
            position operator is returned cartesian coordinates by default
            X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
            r^{\alpha} \vert u_{n {\bf k}} \rangle            
        ''' 
        pass
        # r_mnk = np.zeros((self.nb, self.nb,3), dtype=np.complex128)
        # for i in range(self.nb):
        #     for j in range(self.nb):
        #         r_mnk[i,j] = np.vdot(eigvec[:,i], np.dot(self.pos*eigvec[:,j])) 
        
        # return r_mnk    
    @classmethod
    def _reshape_eigvec(cls, nk, nb, f_kn, eigv, eigvec):
        """
        Reshapes the eigenvectors and eigenvalues from a full matrix to separated valence and conduction bands.

        Parameters
        ----------
        nk : int
            The number of k-points.
        nb : int
            The number of bands.
        f_kn : ndarray
            The occupation number at each k-point and band.
        eigv : ndarray
            The eigenvalues.
        eigvec : ndarray
            The eigenvectors.

        Notes
        -----
        The eigenvectors are reshaped from a full matrix to separated valence and conduction bands.
        The valence (conduction) bands are stored in the first (last) `nv` (`nc`) columns of the eigenvectors.
        The eigenvalues are also separated into valence and conduction bands.
        The occupation number is used to distinguish between valence and conduction bands.
        """
        nv = np.count_nonzero(f_kn[0])
        nc = nb -nv
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
                    eigvc[k,n-nv] = eigv[k,n]
                    eigvecc[k,:,n-nv] = eigvec[k,:,n]
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
        self.ntransitions = ntransitions
        self.T_table = T_table

    @classmethod
    def _get_occupations(cls, nk, nb, eigv, fermie):
        occupations = np.zeros((nk, nb))
        occupations = fermi_dirac(eigv,fermie)
        cls.f_kn = np.real(occupations)
    
    def _build_Umn(self):
        '''A = PDA^\dagger where D is a diagonal matrix but then D = P\daggerA P. The eigenvectors obtained via numpy diagonalization
        should be the columns of P. I checked this and indeed I get P where P is U_nm(k).
        Where |mR_e> = \sum_nk  e^-ikRe U_nmk | nk> . Since in this formalism n is running in the first index I need to tranpose the result.
        I put k on the first index for convenience.
        '''
        Umn = np.zeros(shape=(self.nk, self.nb, self.nb), dtype = np.complex128)
        Umn[:,:, :self.nv] = self.eigvecv
        Umn[:,:, self.nv:self.nb] = self.eigvecc
        Uknm = Umn.transpose(0,2,1)
        self.Uknm = Uknm

    def _get_overlap(self):
        Mmn = np.zeros((self.nb, self.nb,self.nk, self.nk), dtype=np.complex128)
        # here l stands for lambda, just to remember me that there is a small difference between lambda and transition index
        for n in range(self.nb):
            for m in range(self.nb):   
                for ik in range(self.nk):
                    for ikp in range(self.nk):
                        Mmn[n,m,ik, ikp] = np.vdot(self.eigvec[ik,:,n],self.eigvec[ikp,:,m])
        self.Mmn = Mmn

    def write_overlap(self,seedname='wannier90',):
        if (self.Mmn is None):
            self._get_overlap()

        from datetime import datetime
        
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d at %H:%M:%S")
        f_out = open(f'{seedname}.mmn', 'w')
        f_out.write(f'Created on {date_time_string}\n')
        f_out.write(f'\t{self.nb}\t{self.nk}\t{self.nk}\n')  
        (self.kplusq_table, self.kminusq_table) = self.mpgrid.get_kq_tables(self.mpgrid)      
        for ik in range(self.nk):
            for ikp in range(self.nk):
                # +1 is for Fortran counting, assume all Gs are 0 for WanTIBEXOS (to be discussed)
                f_out.write(f'\t{self.kplusq_table[ik,ikp,1]+1}\t{self.kplusq_table[ik,ikp,1]+1}\t{0}\t{0}\t{0}\n')
                for n in range(self.nb):
                    for m in range(self.nb):
                        f_out.write(f'\t{np.real(self.Mmn[m,n,ik,ikp]):.14f}\t{np.imag(self.Mmn[m,n,ik,ikp]):.14f}\n')
        
        f_out.close()