
import numpy as np

HA2EV  = 27.211396132

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