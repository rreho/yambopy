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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def solve_ham(self, k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]], convention: int = 2):
        self.H_k = self.hamilton(k, convention=convention)
        self.nk = int(self.H_k.shape[0]) # number of k-points
        self.nb = int(self.H_k.shape[1]) # number of bands
        # check hermiticity
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
