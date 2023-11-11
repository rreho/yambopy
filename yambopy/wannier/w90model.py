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


class TBMODEL(tbmodels.Model):
    '''
    Class that inherits from tbmodels.Model for TB-model Hamiltonians
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
        self.eigv= np.zeros((self.nk,self.nb))
        self.eigvec = np.zeros((self.nk,self.nb,self.nb))
        for ik in range(self.nk):
            (self.eigv[ik], self.eigvec[ik]) = np.linalg.eigh(self.H_k[ik])
            (self.eigv[ik],self.eigvec[ik]) = self._sort_eig(self.eigv[ik],self.eigvec[ik])
        # transpose to have eig[:,i] associated with eval[i]
        # one could transpose it to have the opposite, for now commented out bcs I don't like it
        #self.eig = self.eig.T
        # sort eigenvectors 
        # To-do should I check for the spin case?
        t1 = time()
        print(f'Diagonalization took {t1-t0:.3f} sec')

    @classmethod
    def get_pos_from_ham(self, lat):
        'get positions from Hamiltonian, first the indices and then cartesian coordinates of hoppings'
        # get tuple of irpos
        self.irpos = np.array(list(self.hop.keys())) 
        self.nrpos = len(self.irpos)
        self.lat = lat
        # pos[i,3] position of i unit cells. Store only the upper triangular matrix
        self.pos = np.dot(self.lat, self.irpos.T).T


    @classmethod
    def get_hlm(cls, k, lat, v, q, c):
        ''' computes light mater interaction hamiltonian
        k is a list of k-point in cartesian coordinates
        '''
        pos = cls.get_pos_from_ham(lat)
        hlm = np.zeros(cls.nb, cls.nb)
        kvecs = np.zeros(cls.nb)
        kvecsx = np.zeros(cls.nb)
        kvecsy = np.zeros(cls.nb)
        kvecsz = np.zeros(self.nb)


        # set kveces
        for i in range(0,len(pos)):
            kvecs

        for ip, p in enumerate(pos):
            hlm = hlm + np.exp(p*k[i])





    @classmethod
    def pos_operator_matrix(cls, eigvec, dir, cartesian = True):
        ''' Computes the position operator along a direction dir at a k-point
            position operator is returned cartesian coordinates by default
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
