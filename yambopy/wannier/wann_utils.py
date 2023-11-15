
import numpy as np

HA2EV  = 27.211396132
BOHR2ANG = 0.52917720859
ANG2BOHR = 1./BOHR2ANG

def fermi_dirac_T(e, T, fermie):
    # atomic units
    kb = 3.1671e-06
    fermi = 1.0/(np.exp(e-fermie)/(kb*T))
    return fermi

def fermi_dirac(e, fermie):
    """ Vectorized Fermi-Dirac function. """
    # Create a boolean mask for conditions
    greater_than_fermie = e > fermie
    less_or_equal_minus_fermie = e <= -fermie

    # Initialize the result array with zeros (default case when e > fermie)
    result = np.zeros_like(e)

    # Apply conditions
    result[less_or_equal_minus_fermie] = 1

    return result

def sort_eig(eigv,eigvec=None):
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