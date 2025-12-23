# Copyright (c) 2023-2025, Claudio Attaccalite,
#                          Myrta Gruening, Anna Romani
# All rights reserved.
#
# This file is part of the yambopy project
# Calculate linear response from real-time calculations (yambo_nl)
# Modified to allow generalisation
#
import numpy as np
from yambopy.units import ha2ev,fs2aut
from yambopy.nl.external_efield import Divide_by_the_Field, Gaussian_centre
from tqdm import tqdm
from math import floor, ceil, factorial
import sys
import os
from abc import ABC,abstractmethod
from yambopy.nl.nl_analysis import Xn_from_signal
#
# Derived class for pulse signal
#    
class Xn_from_pulse(Xn_from_signal):
    #
    def set_defaults(self):
        EFIELDS = ["QSSIN"]
        if self.efield["name"] not in EFIELDS:
            raise ValueError(f"Invalid electric field for frequency mixing analysis. Expected one of: {EFIELDS}")
        for i_n in range(len(self.pumps)):
            if self.pumps[i_n]["name"] != 'none':
                raise ValueError("This analysis is for one monochromatic field only.")
        if self.solver == '':
            self.solver = 'full'
        dim_ = int(1+self.X_order+(self.X_order/2)**2)
        D = (dim_,int(dim_ -0.25))
        self.out_dim = D[self.X_order%2]
        self.T0 = Gaussian_centre(self.efield)
        return
    
    def build_map(self):
        M = 1+2*self.X_order + int(self.X_order*(self.X_order-1)/2) 
        L = (int(self.X_order/2*((self.X_order/2+1))),int(((self.X_order+1)/2)**2))
        I = np.zeros((M,2),dtype=int) 
        j = 0
        I[j,:] = [0,0]
        for n in range(2,N+1,2):
            j += 1
            I[j,:] = [n,floor(n/2)]
        for n in range(1,N+1):
            for m in range(0,ceil(n/2)):
                j += 1
                I[j,:] = [n,m]
                I[j+L[self.X_order%2],:] = [-n,m]
        return I

    def get_sampling(self,idir,ifrq):
        M = 1+2*self.X_order + int(self.X_order*(self.X_order-1)/2) 
        if self.nsamp == -1 or self.nsamp < M:
            self.nsamp = M
        T_period = 2.0 * np.pi / self.freqs[ifrq]
        T_range, out_of_bounds = self.update_time_range(T_period)
        if (out_of_bounds):
            print(f'User range redifined for frequency {self.freqs[ifrq]* ha2ev:.3e} [eV]')
        print(f"Time range: {T_range[0] / fs2aut:.3f} - {T_range[1] / fs2aut:.3f} [fs]")
        i_t_start = int(np.round(T_range[0]/self.T_step)) 
        i_deltaT  = int(np.round(T_period/self.T_step)/self.nsamp)
        T_i = np.array([(i_t_start + i_deltaT * i) * self.T_step - self.efield["initial_time"] for i in range(self.nsamp)])
        S_i = np.array([self.polarization[ifrq][idir,i_t_start + i_deltaT * i] for i in range(self.nsamp)]) 
        return T_i,S_i
        
    def update_time_range(self,T_period): 
        T_range = self.T_urange
        out_of_bounds = False
        if T_range[0] <= 0.0:
            T_range[0] = self.T0 - T_period/2
        T_range[1] = T_range[0] + T_period
        if T_range[1] > self.time[-1]:
            T_range[1] = slef.time[-1]
            T_range[0] = T_range[1] - T_period
            out_of_bound = True
        return T_range, out_of_bounds

    def define_matrix(self,T_i,ifrq):
        dim_ = 1+2*self.X_order + int(self.X_order*(self.X_order-1)/2) 
        W = self.freqs[ifrq]
        M = np.zeros((self.nsamp, dim_), dtype=np.cdouble)
        i_map = self.build_map()
        M[:,0] = 1.0
        for ii in range(1,len(i_map)):
            i_n,i_m = i_map[ii]
            n = abs(i_n)
            M[:,ii] = np.exp(-n*(T_i[:]-self.T_0)**2/(2*self.sigma**2)) #sign
            if (i_n%2 == 0 and i_m == int(i_n/2)):
                M[:,ii] = M[:,ii]
            elif (i_m < ceiling(i_n/2)):
                if (i_n>0):
                    M[:,ii] *= np.exp(-1j * (n-2*i_m)*W * T_i[:]) 
                if (i_n<0):
                    M[:,ii] *= np.exp(1j * (n-2*i_m)*W * T_i[:]) 
        return M
    
    def divide_by_factor(self,f,n,m): # note that this must be generalised and moved to divide by field
        #
        # 1/2 (iEo)^n (-1)^m K (-p*omega; omega, ..., omega, -omega, ..., -omega) => specific for one freq.
        #
        # then K = n!/(m!(n-m)!) * 2**(1-n) if p /= 0, omega \=0 
        #      K = 2**(-n) if p = 0, omega \=0
        #      K = 1 p = 0, omega  =0
        p  = int(n - 2*m)
        factor = 0.5
        if f > 1e-4:
            factor *=2**(-n)
            if p != 0:
                factor *= 2*factorial(n)/factorial(m)/factorial(n-m)
        factor *= (-1)**m*np.power(1.0j*efield['amplitude'],n,dtype=np.cdouble)
        return 1.0/factor
        
    def output_analysis(self,out,to_file=True):
        i_map = self.build_map()
        for ii in range(len(i_map)):
            i_n,i_m = i_map[ii]
            if (i_n < 0): continue
            i_p = int(i_n - 2 * i_m)
            for i_f in range(self.n_runs):
                    out[ii, i_f, :] *= divide_by_factor(self.efields[i_f],self.freqs[i_f],i_n,i_m) 
            out[ii,:,:]*=self.get_Unit_of_Measure(i_n) 
            if (to_file):
                output_file = f'o{self.prefix}.YPy-X_order_{i_n}_{i_p}'
                header = "E[eV] " + " ".join([f"X{i_n}({i_p}w)/Im({d}) X{i_n}({i_p}w)/Re({d})" for d in ('x','y','z')])
                values = np.column_stack((self.freqs * ha2ev, out[ii, :, 0].imag, out[ii, :, 0].real,
                                out[ii, :, 1].imag, out[ii, :, 1].real,
                                out[ii, :, 2].imag, out[ii, :, 2].real))
                np.savetxt(output_file, values, header=header, delimiter=' ', footer="Pulse analysis results") 
            else:
                return (self.freqs, out) 

    def reconstruct_signal(self,out,to_file=True):
        i_map = self.build_map()
        Seff = np.zeros((self.n_runs, 3, len(self.time)), dtype=np.cdouble)
        for i_f in tqdm(range(self.n_runs)):
            for i_d in range(3):
                for ii in range(len(i_map)):
                    i_n,i_m = i_map[ii]
                    if (i_n < 0): continue
                    i_p = int(i_n - 2 * i_m)
                    freq_term = (np.exp(-1j * i_p * self.freqs[i_f] * self.time))*np.exp(-i_n*(self.time-self.T_0)**2/(2*self.sigma**2))
                    Seff[i_f, i_d, :] += out[i_order, i_f, i_d] * freq_term
                    Seff[i_f, i_d, :] += np.conj(out[i_order, i_f, i_d]) * np.conj(freq_term)
        for i_f in tqdm(range(self.n_runs)):
            values = np.column_stack((self.time / fs2aut, Seff[i_f, 0, :].real, Seff[i_f, 1, :].real, Seff[i_f, 2, :].real))
            output_file = f'o{self.prefix}.YamboPy-pol_reconstructed_F{i_f + 1}'
            header="[fs] Px Py Pz"
            if (to_file):
                np.savetxt(output_file, values, header=header, delimiter=' ', footer="Reconstructed signal")
        if (not to_file):
            return values

