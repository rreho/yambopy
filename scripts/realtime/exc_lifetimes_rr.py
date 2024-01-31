#Author : Riccardo Reho
#27/01/2024
#Compute the exciton radiative lifetime from the o*E_sorted file
from yambopy import *
import numpy as np
import re 
cfe = 7.297e-3 # constant fine structure
c = 299.792e16  # speed of ligh Ang/s
hbar = 6.582e-16 # planck const eV*s
#cic = -(0.0904756)*10**(3)#fine structure constant
AU2S =  2.418884326509e-17 # atomic unit of time
def find_max_residual(file_path):
    max_residual = None

    with open(file_path, 'r') as file:
        for line in file:
            if "Maximum Residual Value" in line:
                # Extract the residual value using regex
                match = re.search(r"=\s*([0-9.]+E[+-]?[0-9]+)", line)
                if match:
                    residual_value = float(match.group(1))
                    if max_residual is None or residual_value > max_residual:
                        max_residual = residual_value

    return max_residual
max_res = find_max_residual('o-bse-pl.exc_qpt1_E_sorted')
ylat = YamboLatticeDB.from_db_file(filename='./SAVE/ns.db1')
A_uc =np.linalg.norm(np.cross(ylat.lat[0], ylat.lat[1])) * bohr2ang**2  #area of unit cell
data = np.loadtxt('o-bse-pl.exc_qpt1_E_sorted', usecols=[0,1])
s = 2
nsoc = 1
E = data[:,0]
residual = data[:,1] * max_res * ylat.nkpoints * ylat.lat_vol * bohr2ang**3 / (4*np.pi*nsoc*HA2EV) # multiply because in the file dipoles are normalized and convert to meter, they are already the square modulus
factor1 = ylat.nkpoints*A_uc*hbar
factor2 = 8.0*np.pi*cfe*E[s]/(ha2ev*nsoc)*residual[s]

gamma_s  = factor1/factor2
print('Exciton radiative decay = ', gamma_s, ' [1/s]')
print('Exciton radiative lifetimes = ', 1/gamma_s, ' [s]')
