##
# Authors: MN FP
##

import numpy as np
from yambopy.units import ha2ev
from yambopy.tools.funcs import bose,boltzman_f
from tqdm import tqdm
from yambopy.tools.citations import citation
#from numba import prange#,njit

#@njit(cache=True, nogil=True, parallel=True)
@citation("M. Zanfrognini et al. Phys. Rev. Lett. 131, 206902 (2023)")
def exc_ph_luminescence(ph_temp,ph_energies,exc_energies,exc_dipoles,exc_ph_mat_el,\
                        exc_energies_in=None,exc_temp=None,ph_channels='b',\
                        PL_energy_prefactor='PT',nexc_out='all',nexc_in='all',\
                        emin=0,emax=10,estep=0.01,broad=0.1,broad_0=None,\
                        omega_min=None,res_min=None):
    """
    This function calculates the phonon-assisted satellites (phonon replica) in the 
    luminescence of indirect materials (i.e., phonon-mediated exciton recombination process).

    It uses a perturbation theory derivation that includes the off-diagonal elements
    of the exciton-phonon matrix elements.

    Exciton populations are approximated with a Boltzmann distribution function.

    See:
    - Phys. Rev. Materials 7, 024006 (2023)
    - Phys. Rev. Letters 131, 206902 (2023) Equations from (1) to (4)

    Returns:
    ----------
    * Energy axis in specified range (in eV)
    * Luminescence intensity

    Parameters
    ----------
    ph_temp : float
        Lattice temperature in kelvin
    ph_energies : float ndarray
        Phonon energies in eV [nqpts,nmodes]
    exc_energies : float ndarray
        Exciton energies in eV [nqpts,nexc_out] 
        Typically, YamboExcitonDB.eigenvalues are used
    exc_dipoles : cmplx ndarray
        Exciton dipole matrix elements in a.u. (bohr) 
        * If shape is [nexc_in], assume projected dipoles: typically, YamboExcitonDB.r_residuals are used.
        * If shape is [pol,nexc_in], perform polarization average. This can be produced with exc_dip_pol function in yambopy.bse.excitondipoles
    exc_ph_mat_el : cmplx ndarray
        Exciton-phonon coupling matrix elements in a.u. (hartree) [nqpts,nmodes,nexc_in,nexc_out]
        Typically, computed via exciton_phonon_matelem
    exc_energies_in : float ndarray, optional
        If given, zero-momentum states will be taken from this array. Default is exc_energies.
    exc_temp : float
        If given, effective temperature controlling excitonic population. Default is ph_temp.
    ph_channels : str
        'e'  -> only phonon emission channel
        'a' -> only phonon absorption channel
        'b' -> both channels (default)
    PL_energy_prefactor : str
        'PT' -> prefactor from time-dependent perturbation theory (default)
        'RS' -> prefactor from van Roosbroeck-Shockley relation, w^3 at the photon energy
                (not including refraction index). Note: PRM 7, 024006 Eq. 17 uses w(w-/+2w_q)^2
                for the satellites, which does not follow from RS at the photon energy.
        None or any other str -> no prefactor is used
    nexc_out : int
        Number of finite-q excitons in the sum. Default: all available states.
    nexc_in : int
        Number of optical excitons in the sum. Default: all available states.
    emin, emax, esteps : float
        Laser energy range parameters in eV.
    broad : float
        Broadening parameter in eV.
    broad_0 : float
        Broadening parameter used inside satellite oscillator strengths in eV. Default is broad.
    omega_min : float, optional
        Phonon modes with energy below omega_min (eV) are excluded. Default: None (all modes).
    res_min : float, optional
        Terms with |E_out(q) -/+ omega_q - E_in| < res_min (eV) are excluded from the satellite
        amplitude. These on-shell (double-pole) terms appear in direct-gap materials, where
        finite-q states are nearly resonant with the bright exciton; they belong to the
        zero-phonon line width/shift, not to the replicas. Use together with
        exc_direct_luminescence. Default: None (no cutoff, original formula).
    """
    def get_PL_satellite(W,ph_sign=-1):
        """ Actual calculation of PL satellites at each frequency.
            Uses the same variables as main function.

            W --> laser energy value in hartree
            ph_sign -->
                * -1 : phonon emission
                * +1 : phonon absorption
        """
        # Satellite oscillator strenghts
        ## Dimensions: [nqpts, nmodes, nexc_in, nexc_out]
        satellite_energy = exc_energies[:,None,None,:]-exc_energies_in[None,None,:,None] \
                            + ph_sign*ph_energies[:,:,None,None] + 1j*broad_0
        satellite_energy_Ha = satellite_energy/ha2ev
        satellite_weight = exc_ph_mat_el/satellite_energy_Ha
        if res_min is not None:
            satellite_weight = np.where(np.abs(satellite_energy.real)<res_min, 0., satellite_weight)
    
        # Phonon emission/absorption factor: if em --> ph_occ+1, if abs --> ph_occ
        ph_fac = ph_occ - (ph_sign-1)/2.
    
        # Sum over exc_in and then square (this includes off-diagonal excph SE terms)
        T = np.einsum('ri,qmio->rqmo',exc_dipoles,satellite_weight,optimize=True)
        T = np.sum( np.abs(T)**2, axis=0)/3. # Average over pol. directions as well
        # Include occupation factors
        T = T * exc_occ[:,None,:]*ph_fac[:,:,None]

        # Energy differences [nqpts,nmodes,nexc_out]
        pole_energy = exc_energies[:,None,:]+ph_sign*ph_energies[:,:,None]
        pole_energy_Ha = pole_energy[:,:,:]/ha2ev
        
        # PL energy prefactors
        # Note: with 'PT+RS' as argument both will be considered
        light_fac = 1.
        if 'PT' in PL_energy_prefactor: # [nqpts,nmodes,nexc_out]
            light_fac = light_fac * 1./pole_energy_Ha
        if 'RS' in PL_energy_prefactor: # [nqpts,nmodes,nexc_out]
            light_fac = light_fac * W**3

        # Energy conservation
        delta_funct = 1./((W-pole_energy_Ha)**2.+ broad_Ha**2.)
        E = light_fac*delta_funct
        # Dimensions
        E = E * broad_Ha/np.pi/Nqpts * ha2ev**2

        # Putting everything together
        return np.einsum('qmo,qmo->', E, T, optimize=True) 

    # Checks
    assert exc_energies.shape[0]==ph_energies.shape[0], "Q-point mismatch between excitons and phonons"
    Nqpts = ph_energies.shape[0]
    if exc_energies_in is None: exc_energies_in = exc_energies[0]
    if np.iscomplexobj(exc_energies):    exc_energies    = exc_energies.real
    if np.iscomplexobj(exc_energies_in): exc_energies_in = exc_energies_in.real
    assert ph_channels in ['b', 'e', 'a'], "Allowed values for phonon channels are 'b', 'e', 'a'"
    nexc_out_avail = min(exc_energies.shape[1],exc_ph_mat_el.shape[3])
    nexc_in_avail  = min(len(exc_energies_in),exc_dipoles.shape[-1],exc_ph_mat_el.shape[2])
    if nexc_out =='all': nexc_out = nexc_out_avail
    if nexc_in  =='all': nexc_in  = nexc_in_avail
    assert nexc_out <= nexc_out_avail, "less exciton states than requested (nexc_out)"
    assert nexc_in  <= nexc_in_avail,  "less exciton states than requested (nexc_in)"
    exc_energies    = exc_energies[:,:nexc_out]
    exc_energies_in = exc_energies_in[:nexc_in]
    exc_dipoles     = exc_dipoles[...,:nexc_in]
    # We conj because we need it for photon emission
    exc_ph_mat_el = np.conj( exc_ph_mat_el[...,:nexc_in,:nexc_out] ) 
    assert ph_energies.shape[1]==exc_ph_mat_el.shape[1], "number of modes mismatch between phonon energies and matrix elements"
    if broad_0 is None: broad_0 = broad # This goes in |E1-E2+i*broad_0|^(-2)
    broad = broad/2 # We are using explicit Lorentzian shape
    if exc_temp is None: exc_temp = ph_temp
    # We expect the dipoles to be unpolarized (3,nexc_in). If not, we expect projected dipoles
    # We extend the projected dipoles for compatibility
    if exc_dipoles.ndim==1: exc_dipoles = np.broadcast_to(exc_dipoles, (3, nexc_in)) 
    # Acoustic phonons at q=0: they are already excluded as exc_ph_mat_el[0,0:3,...]=czero
    if omega_min is not None:
        exc_ph_mat_el = exc_ph_mat_el*(ph_energies>=omega_min)[:,:,None,None]

    # Occupation functions
    exc_min_energy = np.min(exc_energies)
    exc_occ = boltzman_f(exc_energies-exc_min_energy,exc_temp)
    ph_occ  = bose(ph_energies,ph_temp)

    # Laser energies
    light_energies = np.arange(emin,emax,estep,dtype=np.float32)
    nfreqs = len(light_energies)
    light_energies_Ha = light_energies/ha2ev#light_energies[None,None,None,:]/ha2ev
    broad_Ha = broad/ha2ev

    # Calculation
    PL_satellites = np.zeros(nfreqs)
    for w in tqdm(range(nfreqs),desc="Luminescence"):
        # Accumulate phonon emission satellites
        if ph_channels=='e' or ph_channels=='b': 
            PL_satellites[w] += get_PL_satellite(light_energies_Ha[w],ph_sign=-1)
        # Accumulate phonon absorption satellites
        if ph_channels=='a' or ph_channels=='b': 
            PL_satellites[w] += get_PL_satellite(light_energies_Ha[w],ph_sign=+1)

    return light_energies, PL_satellites


@citation("M. Zanfrognini et al. Phys. Rev. Lett. 131, 206902 (2023)")
def exc_direct_luminescence(exc_temp,exc_energies_in,exc_dipoles,exc_min_energy=None,\
                            PL_energy_prefactor='PT',nexc_in='all',R_lambda=None,\
                            emin=0,emax=10,estep=0.01,broad=0.1,zpl_depletion=None,max_depletion=0.3):
    """
    Direct (zero-phonon, first-order) luminescence from Q=0 excitons.

        I_0(w) = sum_l |T_l|^2/3 * (1-R_l) * N_l * P(E_l) * L(w-E_l)

    T_l    : exciton dipoles (same as in exc_ph_luminescence)
    N_l    : Boltzmann population exp[-(E_l-E_min)/kT_exc]
    P(E_l) : 'PT' -> 1/E_l (same convention as the satellites), 'RS' -> E_l^3
    L      : Lorentzian with FWHM broad (eV), or per-state widths if broad is an array
    R_l    : optional weight removed from the line, factor (1-R_l) (R_lambda, no cutoff).
             The replica weight (exc_ph_renormalisation, PRM 7, 024006 Eqs. 14-15)
             enforces the satellite sum rule.
    zpl_depletion : optional 1-Z_l from exc_ph_zpl_depletion (same T, cutoffs and eta
             as the satellites). The line of state l is weighted by Z_l = 1-(1-Z_l) if
             1-Z_l <= max_depletion; states with 1-Z_l > max_depletion (second order
             not reliable) are omitted from the direct line and printed.

    Same units and normalisation as exc_ph_luminescence, so the two spectra can be
    summed. To use with direct-gap materials, compute the satellites with res_min
    (on-shell terms removed) and give here the zero-phonon linewidths, e.g. from the
    imaginary part of the exciton-phonon self-energy.
    Important: pass exc_min_energy = min over ALL exciton energies (finite Q included)
    so that populations are consistent with exc_ph_luminescence.

    Parameters
    ----------
    exc_temp : float
        Excitonic temperature in kelvin
    exc_energies_in : float ndarray [nexc_in], Q=0 exciton energies in eV
    exc_dipoles : cmplx ndarray [3,nexc_in] or [nexc_in], bohr
    exc_min_energy : float, optional. Reference energy for populations. Default min(exc_energies_in)
    broad : float or float ndarray [nexc_in], FWHM in eV

    Returns
    ----------
    * Energy axis (eV), direct luminescence intensity
    """
    exc_energies_in = np.real(exc_energies_in)
    if nexc_in=='all': nexc_in = min(len(exc_energies_in),exc_dipoles.shape[-1])
    E = exc_energies_in[:nexc_in]
    D = exc_dipoles[...,:nexc_in]
    if D.ndim==1: D = np.broadcast_to(D,(3,nexc_in))
    if exc_min_energy is None: exc_min_energy = np.min(E)
    occ  = boltzman_f(E-exc_min_energy,exc_temp)
    osc  = np.sum(np.abs(D)**2,axis=0)/3.          # polarization average
    E_Ha = E/ha2ev
    light_fac = np.ones(nexc_in)
    if 'PT' in PL_energy_prefactor: light_fac = light_fac/E_Ha
    if 'RS' in PL_energy_prefactor: light_fac = light_fac*E_Ha**3
    hwhm_Ha = np.broadcast_to(np.asarray(broad)/2.,(nexc_in,))/ha2ev
    light_energies = np.arange(emin,emax,estep,dtype=np.float32)
    W = light_energies[:,None]/ha2ev
    lor = hwhm_Ha[None,:]/np.pi/((W-E_Ha[None,:])**2+hwhm_Ha[None,:]**2)
    # same dimensional factor as the satellites (ha2ev**2, no 1/Nq: Q=0 only)
    if R_lambda is not None: osc = osc*(1.-np.asarray(R_lambda)[:nexc_in])
    if zpl_depletion is not None:
        assert R_lambda is None, "give either R_lambda or zpl_depletion"
        dZ = np.asarray(zpl_depletion)[:nexc_in]
        drop = dZ > max_depletion
        if np.any(drop):
            print('[direct PL] omitted states (1-Z > %.2f): %s' % (max_depletion, ', '.join(
                  'S%d E=%.4f eV 1-Z=%.3g' % (l+1, E[l], dZ[l]) for l in np.where(drop)[0])))
        osc = np.where(drop, 0., osc*(1.-dZ))
    PL_direct = np.einsum('wl,l->w',lor,osc*occ*light_fac)*ha2ev**2
    return light_energies, PL_direct


def _excph_resonances(ph_temp,ph_energies,exc_energies,exc_ph_mat_el,exc_energies_in,
                      nexc_out,nexc_in,omega_min):
    """ |G|^2 (eV^2), n, E_l-E_qb (eV), w (eV), each broadcast to [q,m,l,b]. """
    exc_energies = np.real(exc_energies)
    if exc_energies_in is None: exc_energies_in = exc_energies[0]
    exc_energies_in = np.real(exc_energies_in)
    if nexc_out=='all': nexc_out = min(exc_energies.shape[1],exc_ph_mat_el.shape[3])
    if nexc_in =='all': nexc_in  = min(len(exc_energies_in),exc_ph_mat_el.shape[2])
    G2 = np.abs(exc_ph_mat_el[...,:nexc_in,:nexc_out])**2*ha2ev**2   # [q,m,l,b] eV^2
    if omega_min is not None: G2 = G2*(ph_energies>=omega_min)[:,:,None,None]
    n  = bose(ph_energies,ph_temp)[:,:,None,None]
    dE = exc_energies_in[None,None,:nexc_in,None]-exc_energies[:,None,None,:nexc_out]
    return G2, n, dE, ph_energies[:,:,None,None]


def exc_ph_renormalisation(ph_temp,ph_energies,exc_energies,exc_ph_mat_el,exc_energies_in=None,\
                           nexc_out='all',nexc_in='all',omega_min=None,res_min=None,broad_0=0.):
    """
    Replica weight of each Q=0 exciton l: total weight of its phonon satellites
    (diagonal approximation, PRM 7, 024006 Eq. 15 with 1/N_q):

        R_l = 1/N_q sum_{q,m,b} |G_m[q,l,b]|^2 [ (n_qm+1)/|E_l-E_qb+w_qm+i eta|^2 + n_qm/|E_l-E_qb-w_qm+i eta|^2 ]

    Emission pairing: the replica at E_qb-w (state qb emits phonon + photon) carries
    n+1 and the denominator E_qb-w-E_l. This is the satellite sum rule, NOT the
    zero-phonon-line depletion 1-Z_l (exc_ph_zpl_depletion: n <-> n+1 swapped).
    Same inputs, units and cutoffs (omega_min, res_min, broad_0=eta in eV) as
    exc_ph_luminescence, so (1-R_l) + sum_sat = 1 holds term by term with the same broad_0.
    Meaningful only for R_l << 1. Returns R [nexc_in].
    Alias: exc_ph_replica_weight.
    """
    G2, n, dE, w = _excph_resonances(ph_temp,ph_energies,exc_energies,exc_ph_mat_el,
                                     exc_energies_in,nexc_out,nexc_in,omega_min)
    R = 0.
    for sgn,occ in ((+1,n+1.),(-1,n)):
        den = dE+sgn*w
        inv = np.where(np.abs(den)<(res_min if res_min is not None else 0.),0.,1./(den**2+broad_0**2+1e-300))
        R = R + np.sum(G2*occ*inv,axis=(0,1,3))
    R = R/ph_energies.shape[0]
    if np.max(R)>0.3: print(f"[WARNING] max R = {np.max(R):.2f}: second-order weight transfer not reliable (R should be << 1)")
    return R

exc_ph_replica_weight = exc_ph_renormalisation


def exc_ph_zpl_depletion(ph_temp,ph_energies,exc_energies,exc_ph_mat_el,exc_energies_in=None,\
                         nexc_out='all',nexc_in='all',omega_min=None,res_min=None,broad_0=0.):
    """
    Zero-phonon-line depletion 1-Z_l of each Q=0 exciton l from the Fan-Migdal
    exciton-phonon self-energy (excitons unoccupied, f=0 in the (n+1-f), (n+f) factors):

        Sigma_l(w) = 1/N_q sum_{q,m,b} |G_m[q,l,b]|^2 [ (n_qm+1)/(w-E_qb-w_qm+i eta) + n_qm/(w-E_qb+w_qm+i eta) ]

        1-Z_l = -dRe Sigma_l/dw |_{E_l}
              = 1/N_q sum |G|^2 [ (n+1) (x_-^2-eta^2)/(x_-^2+eta^2)^2 + n (x_+^2-eta^2)/(x_+^2+eta^2)^2 ],
        x_-+ = E_l-E_qb-+w_qm   (eta -> 0: (n+1)/x_-^2 + n/x_+^2)

    i.e. (n+1) pairs with E_l-E_qb-w (l emits a phonon), n with E_l-E_qb+w.
    Inputs and cutoffs as exc_ph_renormalisation (terms with |x| < res_min dropped).
    The direct line carries Z_l = 1-(1-Z_l): pass the result as R_lambda to
    exc_direct_luminescence. Meaningful only for 1-Z_l << 1. Returns 1-Z [nexc_in].
    """
    G2, n, dE, w = _excph_resonances(ph_temp,ph_energies,exc_energies,exc_ph_mat_el,
                                     exc_energies_in,nexc_out,nexc_in,omega_min)
    dZ = 0.
    for sgn,occ in ((-1,n+1.),(+1,n)):
        x2 = (dE+sgn*w)**2
        f = np.where(np.sqrt(x2)<(res_min if res_min is not None else 0.),0.,
                     (x2-broad_0**2)/((x2+broad_0**2)**2+1e-300))
        dZ = dZ + np.sum(G2*occ*f,axis=(0,1,3))
    dZ = dZ/ph_energies.shape[0]
    if np.max(dZ)>0.3: print(f"[WARNING] max 1-Z = {np.max(dZ):.2f}: second-order depletion not reliable (should be << 1)")
    return dZ
