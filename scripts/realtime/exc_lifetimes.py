import numpy as np
import argparse
import sys
import re

HA2EV=27.2113834
SPEED_OF_LIGHT = 137.03599911 # a.u.
AU2SEC = 2.418884326505E-17 # Tau = AU2SEC sec
KB = 8.617330337217213e-05/HA2EV # Ha/K
m_e = 0.510998950e+06 # electron mass in eV


parser = argparse.ArgumentParser()
parser.add_argument('filesetup',  nargs='?', default=None, help='r_setup file')
parser.add_argument('filesorted',  nargs='?', default=None, help='o-*_E_sorted file')
#parser.add_argument('--state', type=int, default=0, help='Index of the exciton state to analyse')
parser.add_argument("--statelist",nargs="*",type=int,default=[1], help='List of separate exciton states to analyse. Use ase --statelist N1 N2 N3 .. NN')
parser.add_argument('--degen_step',  type=float, default=0.001, help='Maximum energy separation of two degenerate states (eV)')
parser.add_argument('--T',  type=float, default=0.0, help='Temperature (K)')
parser.add_argument('--MS',  type=float, default=0.0, help='Exciton effective mass (m_e)')
parser.add_argument('--fileoutput',  type=str, default='./output.dat', help='Output file. If not specified, a default file is created as ./output.dat')
args = parser.parse_args()

filesetup = args.filesetup
filesorted = args.filesorted
fileoutput = args.fileoutput
degen_dE = args.degen_step
statelist = args.statelist
#state_internal = state-1
T = args.T
MS = args.MS


def get_vector_from_line(line):
    # Split the line and filter out non-numeric parts
    parts = line.split()
    return np.array([float(part) for part in parts if is_float(part)])

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

with open(filesetup, 'r') as setupfile:
    lines = setupfile.readlines()
    gauge = None
    for i, line in enumerate(lines):
        if 'Spin polarizations' in line:
            spin_degen = float(line.strip().split()[-1])
        elif 'BZ  Q-points' in line:
            Nk = float(line.strip().split()[-1])  # Number of k points in full BZ
        elif 'A[ 1 ]:' in line and i + 1 < len(lines):
            a1 = get_vector_from_line(lines[i + 1])
        elif 'A[ 2 ]:' in line and i + 1 < len(lines):
            a2 = get_vector_from_line(lines[i + 1])
        elif 'A[ 3 ]:' in line and i + 1 < len(lines):
            a3 = get_vector_from_line(lines[i + 1])
        elif re.search(r"Global Gauge\s*:\s*(velocity|length)", line):
            gauge_match = re.search(r"Global Gauge\s*:\s*(velocity|length)", line)
            if gauge_match:
                gauge = gauge_match.group(1)

#unit cell volume           
Omega = np.dot(a1,np.cross(a2,a3)) # a.u.^3
#unit cell area
A = np.cross(a1,a2)[2]


# read E_sorted file
excE,excI = np.loadtxt(filesorted,usecols=(0,1),unpack=True)
# get maximum residual value
with open(filesorted,'r') as setupfile:
    for line in setupfile.readlines():
        if 'Maximum Residual Value ' in line:
            MaxRes = float(line.strip().split()[-1])
# get residual list
excI *= MaxRes


tau0_tot = np.zeros(len(statelist))
tauT_tot = np.zeros(len(statelist))
merged_states = np.empty(len(statelist), dtype='object')

# cycle on states if many in input list
for l,state in enumerate(statelist):
    state_internal = state-1
    
    # find states within degen_step window from state
    mesk = np.logical_and(excE>=excE[state_internal]-degen_dE,excE<=excE[state_internal]+degen_dE)
    states_idx = np.where(mesk==True)[0]
    #q0_norm factor
    q0_norm = 1e-5
    # compute gamma
    gamma0 = 0
    gammaT = 0
    for i,st in enumerate(states_idx):
        ES = excE[st]/HA2EV
        if (gauge == 'length'):
            muS2 = excI[st]*Omega*Nk/(4.*np.pi*HA2EV) # get exciton dipole from residual
        elif (gauge == 'velocity'):
            muS2 = excI[st]*Omega*Nk/(4.*np.pi*HA2EV)*q0_norm**2/(ES**2)
        gg = 4.*np.pi*ES*(muS2/Nk)/(A*SPEED_OF_LIGHT)
        gamma0 += gg
        if MS*T > 0:
            alpha = 4.*ES**2/(3.*2.*MS*SPEED_OF_LIGHT**2*KB*T)
            gammaT += gg/alpha

    # compute tau
    tau0_tot[l] = AU2SEC/gamma0
    if MS*T > 0:
        tauT_tot[l] = AU2SEC/gammaT
    else:
        tauT_tot[l] = 0

    merged_states[l]='{}<->{}'.format(min(states_idx)+1,max(states_idx)+1)
    
with open(fileoutput,'w') as fileout:
    fileout.write("#  Exciton radiative lifetime\n")
    fileout.write("#\n")
    fileout.write("#  Effective mass = {} m_e\n".format(MS))
    fileout.write("#  Temperature = {} K\n".format(T))
    fileout.write(f'#  Global gauge = {gauge}\n')
    fileout.write("#  Energy degeneration step = {} eV\n".format(degen_dE))
    fileout.write("#\n")
    fileout.write("#{:>5} {:>15} {:>20} {:>20} {:>20}\n".format('State','Energy (eV)','tau_0 (s)','tau_T (s)','Merged states'))
    for i,state in enumerate(statelist):
        if MS*T > 0: 
            fileout.write("{:>6} {:>15} {:>20.8E} {:>20.8E} {:>20}\n".format(state,excE[state-1],tau0_tot[i],tauT_tot[i],merged_states[i]))
        else:
            fileout.write("{:>6} {:>15} {:>20.8E} {:>20} {:>20}\n".format(state,excE[state-1],tau0_tot[i],'--',merged_states[i]))



