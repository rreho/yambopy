import sys
import numpy as np
import scipy as special
import math  

def get_Efield_w(freqs,efield):
    if efield["name"] == "DELTA":
        efield_w=efield["amplitude"]*np.exp(1j*freqs[:]*efield["initial_time"])
    else:
        print("Fields different from Delta function not implemented yet")
        sys.exit(0)

    return efield_w

    
def Divide_by_the_Field(efield,order):
    
    if efield['name']=='SIN' or efield['name']=='SOFTSIN':
        if order !=0:
            divide_by_field=np.power(-2.0*1.0j/efield['amplitude'],order,dtype=np.cdouble)
        elif order==0:
            divide_by_field=4.0/np.power(efield['amplitude'],2.0,dtype=np.cdouble)
        # elif efield['name'] == 'QSSIN':  ! note that in class Xn_from_pulse there is a factor     
        #
        # This part of code was copied from ypp but it was never used
        #
        # sigma=efield['width']
        # W_0=efield['freq_range'][0]
        # T_0= np.pi/W_0*float(round(W_0/np.pi*3.*sigma))
        # T = 2*np.pi/W_0
        # E_w= math.sqrt(np.pi/2)*sigma*np.exp(-1j*W_0*T_0)*(special.erf((T-T_0)/math.sqrt(2.0)/sigma)+special.erf(T_0/math.sqrt(2.0)/sigma))
        
        # if order!=0:
        #     divide_by_field = (-2.0*1.0j/(E_w*efield['amplitude']))**order
        # elif order==0:
        #     divide_by_field = 4.0/(E_w*efield['amplitude']*np.conj(E_w))
    else:
        raise ValueError("Electric field not implemented in Divide_by_the_Field!")

    return divide_by_field

def Gaussian_centre(efield):
    ratio=np.pi/efield['freq_range'][0]
    sigma=efield["damping"]/(2.0*(2.0*np.log(2.0))**0.5)
    return ratio * float(round(1.0 /ratio * sigma *efield["field_peak"]))

