# Copyright (c) 2023-2025, Claudio Attaccalite,
#                          Myrta Gruening
# All rights reserved.
#
# This file is part of the yambopy project
# Calculate linear response from real-time calculations (yambo_nl)
# Modified to allow generalisation
#
import numpy as np
from yambopy.units import ha2ev,fs2aut, Junit, EFunit,SVCMm12VMm1,AU2VMm1
from yambopy.nl.external_efield import Divide_by_the_Field
from tqdm import tqdm
import sys
import os
from abc import ABC,abstractmethod
#
#
# Template class
#
class Xn_from_signal(ABC):
    def __init__(self,nldb,X_order=4,T_range=[-1, -1],l_out_current=False,nsamp=-1,samp_mod='',solver='',tol=1e-10): # note I need to add user opportunity to set time-range  
        self.time = nldb.IO_TIME_points # Time series
        self.T_step =self.time[1] - self.time[0]     # Time step of the simulation
        self.T_deph =12.0/nldb.NL_damping 
        self.efield = nldb.Efield[0] # External field of the first run
        self.pumps = nldb.Efield_general[1:] 
        self.efields = nldb.Efield
        self.n_runs = len(nldb.Polarization)     # Number of external laser frequencies
        self.polarization = nldb.Polarization     # Array of polarizations for each laser frequency
        self.current     =nldb.Current # Array of currents for each laser frequency
        self.l_eval_current=nldb.l_eval_CURRENT
        self.l_out_current = l_out_current and self.l_eval_current
        self.X_order = X_order
        SOLVE_MODES = ['', 'stnd', 'lstsq', 'svd']
        if solver not in SOLVE_MODES: 
            raise ValueError("Invalid solver mode. Expected one of: %s" % SOLVE_MODES)
        self.solver = solver
        SAMP_MODES = {'','linear', 'log', 'random'}    
        if samp_mod not in SAMP_MODES:
            raise ValueError(f"Invalid sampling mode. Expected one of: {SAMP_MODES}")
        self.samp_mod = samp_mod
        self.nsamp = nsamp
        self.T_urange = T_range 
        self.freqs = np.array([efield["freq_range"][0] for efield in nldb.Efield], dtype=np.double)     # Harmonic frequencies
        self.prefix = f'-{nldb.calc}' if nldb.calc != 'SAVE' else ''
        self.out_dim = 0
        self.tol = tol
        super().__init__()
        
    def __str__(self): 
        """
        Print info of the class
        """
        s="\n * * *  Xn from signal class  * * * \n\n"
        s+="Max time: "+str(self.time[-1])+"\n"
        s+="Time step : "+str(self.T_step)+"\n"
        s+="Type Efield    : "+str(self.efield["name"])+"\n"
        s+="Number of runs   : "+str(self.n_runs)+"\n"
        s+="Current evaluated    : "+str(self.l_eval_current)+"\n"
        s+="Max harmonic order   : "+str(self.X_order)+"\n"
        if self.samp_mod != '':
            s+="Sampling mode    : "+str(self.samp_mode)+"\n"
        if self.solver != '':
            s+="Solver           : "+str(self.solver)+"\n"
        if self.nsamp > 0:
            s+="Sampling points  : "+str(self.nsamp)+"\n"
        if self.T_urange!=[-1, -1]:
            s+="User time range      : "+str(self.T_urange)+"\n"
        s+="Frequency range: ["+str(self.freqs[0])+","+str(self.freqs[-1])+"] [au] \n"
        if  self.l_out_current:
            s+="Output is set to current."
        else:
            s+="Output is set to polarization."
        return s
    
    @abstractmethod
    def set_defaults(self):
        pass

    @abstractmethod
    def get_sampling(self,idir,ifrq):
        pass
    
    @abstractmethod
    def define_matrix(self,samp,ifrq):
        pass

    @abstractmethod
    def update_time_range(self):
        pass

    @abstractmethod
    def get_Unit_of_Measure(self,i_order):
        pass
    
    def solve_lin_system(self,mat,samp):
        mat_dim = int(mat.shape[1])
        out=np.zeros(mat_dim,dtype=np.cdouble)
        if self.solver=="stdn" and ((not IsSquare(mat)) or (not IsWellDefined(mat))):
            print('WARNING: solver changed to least square')
            self.solver = "lstsq"
        if self.solver=="stnd":
            out = np.linalg.solve(mat,samp)
        if self.solver=="lstsq":
            out = np.linalg.lstsq(mat,samp,rcond=self.tol)[0]
        if self.solver=="svd":
            inv = np.linalg.pinv(mat,rcond=self.tol)
            for i_n in range(mat_dim):
                out[i_n]=out[i_n]+np.sum(inv[i_n,:]*samp[:])
        return out
    
    def perform_analysis(self):
        _ = self.set_defaults()
        out = np.zeros((self.out_dim, self.n_runs, 3), dtype=np.cdouble) 
        for i_f in tqdm(range(self.n_runs)):
            for i_d in range(3):
                samp_time, samp_sig= self.get_sampling(i_d,i_f)
                matrix = self.define_matrix(samp_time,i_f)
                raw = self.solve_lin_system(matrix,samp_sig)
                out[:, i_f, i_d] = raw[:self.out_dim]
        return out

    def get_Unit_of_Measure(self,i_order): # not sure if this is a general or specific method - let it here for the moment
        linear = 1.0
        ratio = SVCMm12VMm1 / AU2VMm1 # is there a better way than this?
        if self.l_out_current:
            linear = Junit/EFunit
            ratio = 1.0/EFunit
        if i_order == 0:
            return np.power(ratio, 1, dtype=np.float64)*linear
        return np.power(ratio, i_order - 1, dtype=np.float64)*linear

    @abstractmethod
    def output_analysis(self,out,to_file):
        pass

    @abstractmethod
    def reconstruct_signal(self,out,to_file):
        pass
### some maths auxiliary functions
    def IsSquare(m):
        return m.shape[0] == m.shape[1]

    def IsWellConditioned(m): # with this I am trying to avoid inverting a matrix
        return np.linalg.cond(m) < 1/sys.float_info.epsilon
