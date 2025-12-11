# Copyright (c) 2023-2025, Claudio Attaccalite,Mike Nico Pionteck
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
import itertools
from abc import ABC,abstractmethod
from yambopy.nl.nl_analysis import Xn_from_signal
#
#
#
# Derived class for frequency-mixed signal
#    
class Xn_from_freqmix(Xn_from_signal):
    #
    def set_defaults(self):
        self.l_out_current = False # there is no implementation for current yet though it can be easily introduced
        if self.solver == '':
            self.solver = 'svd'
        if self.samp_mod == '':
            self.samp_mod = 'log'
        EFIELDS = ["SIN","SOFTSIN"]
        if self.efield["name"] not in EFIELDS:
            raise ValueError(f"Invalid electric field for frequency mixing analysis. Expected one of: {EFIELDS}")
        EFIELDS = ["","SIN","SOFTSIN"]
        if self.pumps[0]["name"] not in EFIELDS:
            raise ValueError(f"Invalid pump field for frequency mixing analysis. Expected one of: {EFIELDS}")
        self.pump_freq = 0.0
        if self.pumps[0]["name"] != "none":
            self.pump_freq = self.pumps[0]["freq_range"][0]
        if len(self.pumps)>1 and self.pumps[1]["name"] != "none":
            raise ValueError("Only mixing of two fields implemented.")
        if isinstance (self.X_order,int): # for frequency mixing it expects 2 orders, if one is given the same is given
            self.X_order = [self.X_order,self.X_order]
            if self.pump_freq == 0.0:
                self.X_order[1] = 0
        self.out_dim = (2*self.X_order[0] + 1)*(2*self.X_order[1] + 1)
    #
    def update_time_range(self):
        T_range = self.T_urange
        if T_range[0] <= 0.0:
            T_range[0] = self.T_deph
        if T_range[1] <= 0.0:
            T_range[1]=self.time[-1]
        return T_range
    #    
    def get_sampling(self,idir,ifrq):
        samp_order = self.out_dim
        if self.nsamp == -1 or self.nsamp < samp_order:
            self.nsamp = int(samp_order**2)
        #
        T_range = self.update_time_range()
        i_t_start = int(np.round( T_range[0] / self.T_step)) 
        i_deltaT  = int(np.round((T_range[1]-T_range[0])/self.T_step)/self.nsamp)
        if self.samp_mod=='linear':
            T_i = (i_t_start + i_deltaT * np.arange(self.nsamp))*self.T_step
            P_i = self.polarization[ifrq][idir,i_t_start + i_deltaT * np.arange(self.nsamp)]
        elif self.samp_mod=='log':
            T_i = np.geomspace(i_t_start * self.T_step, T_range[1], self.nsamp, endpoint=False)
            i_t = np.array(np.round(T_i/self.T_step),dtype=int)
            T_i = i_t*self.T_step
            P_i = np.array([self.polarization[ifrq][idir,i] for i in i_t])
        elif self.samp_mod=='random':
            T_i = np.random.uniform(i_t_start * self.T_step, T_range[1], self.nsamp)
            P_i = np.array([self.polarization[ifrq][idir,int(np.round(t / self.T_step))] for t in T_i])
        return T_i,P_i

    def define_matrix(self,T_i,ifrq):
        NX,MX = self.X_order[:]
        W1 = self.freqs[ifrq]
        W2 = self.pump_freq
        M_size = self.out_dim
        M = np.zeros((self.nsamp, M_size), dtype=np.cdouble)
        for i_t in range(self.nsamp):
            for i_c,(i_n,i_m) in enumerate(itertools.product(range(-NX, NX+1),range(-MX, MX+1))):
                M[i_t, i_c] = np.exp(-1j * (i_n*W1+i_m*W2) * T_i[i_t],dtype=np.cdouble)
        return M

    def output_analysis(self,out,to_file=True):
        #
        NX,MX = self.X_order[:]
        #
        for i_f in range(self.n_runs):
            for i_c,(i_n,i_m) in enumerate(itertools.product(range(-NX, NX+1),range(-MX, MX+1))):
                field_fac = 1.0
                if i_n !=0: field_fac *= Divide_by_the_Field(self.efields[i_f],abs(i_n)) #check this
                if self.pump_freq !=0:
                    if i_m !=0: field_fac *= Divide_by_the_Field(self.pumps[0],abs(i_m)) #check this! what is nldb.Efield2?
                out[i_c,i_f,:] *= field_fac
                out[i_c,i_f,:] *= self.get_Unit_of_Measure(abs(i_n)+abs(i_m))
        if (to_file):
            for i_c,(i_n,i_m) in enumerate(itertools.product(range(-NX, NX+1),range(-MX, MX+1))):
                output_file = f'o{self.prefix}.YamboPy-X_probe_order_{i_n}_{i_m}'
                iorder = (i_n-i_m,i_n,i_m)
                header = "E[eV] " + " ".join([f"X{iorder}/Im_{d} X{iorder}/Re_{d}" for d in ('x','y','z')])
                values = np.column_stack((self.freqs * ha2ev, out[i_c, :, 0].imag, out[i_c, :, 0].real,
                        out[i_c, :, 1].imag, out[i_c, :, 1].real,
                        out[i_c, :, 2].imag, out[i_c, :, 2].real))
                np.savetxt(output_file, values, header=header, delimiter=' ', footer=f"Frequency mixing analysis with pump frequency {self.pump_freq*ha2ev} eV") 
        else:
            return (self.freqs, out)

    def reconstruct_signal(self,out,to_file=True):
        Seff = np.zeros((self.n_runs, 3, len(self.time)), dtype=np.cdouble)
        for i_f in tqdm(range(self.n_runs)):
            for i_d in range(3):
                for i_c,(i_n,i_m) in enumerate(itertools.product(range(-NX, NX+1),range(-MX, MX+1))):
                    freq_term = np.exp(-1j * (i_n * self.freqs[i_f] + i_m*self.pump_freq) * self.time)
                    Seff[i_f, i_d, :] += out[i_c, i_f, i_d] * freq_term
        if (to_file):
            for i_f in tqdm(range(self.n_runs)):
                values = np.column_stack((self.time / fs2aut, Seff[i_f, 0, :].real, Seff[i_f, 1, :].real, Seff[i_f, 2, :].real))
                output_file = f'o{self.prefix}.YamboPy-pol_reconstructed_F{i_f + 1}'
                header="[fs] Px Py Pz"
                np.savetxt(output_file, values, header=header, delimiter=' ', footer="Reconstructed signal")
        else:
            return values

        #def spike_correction(X_eff):# Response function to check for spike
        #    return 'Not implemented yet'

        #def residuals_func(x):
        #    return 'Not implemented yet'
