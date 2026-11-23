# License-Identifier: GPL
#
# Copyright (C) 2026 University of Luxembourg
#
# Authors: MN RR
#   MN: original implementation (PhdScripts/exph/raman.py, compute_Raman_oneph_exc)
#   RR: port to yambopy
#
# This file is part of the yambopy project
#
import numpy as np
from yambopy.units import ha2ev, ev2cm1
from yambopy.tools.funcs import bose
from yambopy.tools.citations import citation

@citation("S. Reichardt and L. Wirtz, Sci. Adv. 6, eabb5915 (2020)")
def exc_ph_raman(ph_energies, exc_energies, exc_dipoles, exc_ph_mat_el, laser_energies,
                 broadening=0.1, cell_vol=1.0, nkpts=1, ph_fre_th=5.0, npol=3):
    """
    One-phonon (Stokes) excitonic resonant Raman tensor at q=0.

    Port of `compute_Raman_oneph_exc` (PhdScripts, M. Nalabothula), with the
    gauge-invariant anti-resonant vertex (see below). Only the two time orderings with the phonon vertex between the two photon
    vertices are kept (resonant + fully anti-resonant), TDA excitons at Q=0.

    In the code notation (Hartree atomic units inside):

        w   = laser energy,  W_m = phonon energy of mode m,  eta = broadening/2
        D   = exc_dipoles[:npol]            (photon-emission dipoles, bohr)
        G_m = exc_ph_mat_el[m, S', S]       = <S|dV_m|S'>  (exc_in=S', exc_out=S, Ha)

        R[w,m,i,j] = 1/(nkpts*sqrt(cell_vol)) * sqrt(|w-W_m|/w) * sum_{S,S'} [
              conj(D_iS) conj(G_m[S',S]) D_jS' / ((w - E_S + i eta)(w - E_S' - W_m + i eta))
            + D_iS  conj(G_m[S,S']) conj(D_jS') / ((w + E_S - i eta)(w + E_S' - W_m - i eta)) ]

    Both terms carry the phonon-emission vertex <B|dV_m^+|A> = conj(<A|dV_m|B>) for the
    exciton transition A -> B (resonant: S -> S', anti-resonant: S' -> S), so R only
    changes by a global phase under e_m -> e^{i phi} e_m. The original source has G_m[S',S]
    in the anti-resonant term, equal to conj(G_m[S,S']) only for hermitian G_m (real
    Gamma eigenvectors). See tutorial/raman/MoTe2/NOTES_antiresonant.md.

    i: incoming, j: scattered polarization. No Bose factor and no w*w_s^3 prefactor
    are included in R (as in the source); see `exc_ph_raman_intensity`.
    Modes with |W_m| <= ph_fre_th (cm-1) get R=0.
    The photon prefactor sqrt(|w-W_m|/w) is kept exactly as in the source; it only
    rescales profiles vs laser energy, not mode ratios at fixed w.

    Normalization: the source uses un-normalised exciton dipoles
    D = sum_kcv A^S_kcv d_kcv together with nkpts = N_k. yambopy's `exc_dipoles_pol`
    already includes 1/sqrt(N_k), so with those dipoles use nkpts=1 (default).

    Parameters
    ----------
    ph_energies : float ndarray [nmodes] or [nq,nmodes] (eV). If 2D, q-index 0 (Gamma) is used.
    exc_energies : float ndarray [nexc] (eV), Q=0 exciton energies.
    exc_dipoles : complex ndarray [3,nexc] (bohr), e.g. from yambopy.bse.excitondipoles.exc_dipoles_pol
    exc_ph_mat_el : complex ndarray [nmodes,nexc_in,nexc_out] or [nq,nmodes,nexc_in,nexc_out] (Ha),
        as returned by exciton_phonon_matelem. If 4D, q-index 0 (Gamma) is used.
    laser_energies : float or float ndarray [nlaser] (eV)
    broadening : float, eV (FWHM, the imaginary part of E_S is broadening/2)
    cell_vol : float, unit cell volume (bohr^3)
    nkpts : int, k-point normalization (see above)
    ph_fre_th : float, phonon threshold (cm-1)
    npol : int, number of polarization components used (<=3)

    Returns
    -------
    R : complex ndarray [nlaser, nmodes, 3, 3]
    """
    ph = np.asarray(ph_energies, dtype=float)
    G = np.asarray(exc_ph_mat_el)
    if ph.ndim == 2: ph = ph[0]
    if G.ndim == 4: G = G[0]
    nmodes = G.shape[0]
    assert len(ph) == nmodes, "number of modes mismatch between phonon energies and matrix elements"
    nexc = min(G.shape[1], G.shape[2], len(exc_energies), exc_dipoles.shape[-1])
    G = G[:, :nexc, :nexc]

    w = np.atleast_1d(np.asarray(laser_energies, dtype=float)) / ha2ev
    eta = broadening / ha2ev / 2.0
    Om = ph / ha2ev
    th = ph_fre_th / ev2cm1 / ha2ev
    ram_fac = 1.0 / nkpts / np.sqrt(cell_vol)

    d = np.conj(np.asarray(exc_dipoles)[:npol, :nexc]).astype(np.complex128)  # absorption dipoles
    E = np.asarray(exc_energies[:nexc]).real / ha2ev - 1j * eta                # complex energies
    W = w[:, None, None]
    x_res = d[None] / (W - E)              # [nlaser,p,S]
    x_ares = np.conj(d)[None] / (W + E)

    R = np.zeros((len(w), nmodes, 3, 3), dtype=np.complex128)
    for m in range(nmodes):
        if abs(Om[m]) <= th: continue
        y_res = np.conj(d)[None] / (W - E - Om[m])   # [nlaser,p',S']
        y_ares = d[None] / (W + E - Om[m])
        GmT = G[m].T.astype(np.complex128)          # GmT[S,S'] = G_m[S',S]
        # resonant:      sum_{S,S'} x_res[p,S] conj(G[S',S]) y_res[p',S']
        tmp = (x_res.reshape(-1, nexc) @ np.conj(GmT)).reshape(x_res.shape)
        term1 = np.einsum('wpb,wqb->wpq', tmp, y_res)
        # anti-resonant: sum_{S,S'} x_ares[p,S] conj(G[S,S']) y_ares[p',S']
        tmp = (x_ares.reshape(-1, nexc) @ np.conj(G[m]).astype(np.complex128)).reshape(x_ares.shape)
        term2 = np.einsum('wpb,wqb->wpq', tmp, y_ares)
        scale = np.sqrt(np.abs(w - Om[m]) / w) * ram_fac
        R[:, m, :npol, :npol] = (term1 + term2) * scale[:, None, None]
    return R


def exc_ph_raman_intensity(raman_tensor, e_in=None, e_out=None, ph_energies=None, ph_temp=None):
    """
    Raman intensities from the tensor R[nlaser,nmodes,3,3] of `exc_ph_raman`.

    * e_in is None : I = sum_ij |R_ij|^2  (unpolarized sum, as in PhdScripts raman_program.py)
    * e_in given   : I = |sum_ij e_in_i R_ij conj(e_out_j)|^2 (e_out defaults to e_in;
                     e.g. e_in=[1,0,0] gives the backscattering XX configuration |R_xx|^2)
    * ph_temp (K) given : multiply by the Stokes Bose factor n(W_m,T)+1 (needs ph_energies in eV).
      Default None: no Bose factor (as in the source).

    Returns I [nlaser, nmodes]
    """
    R = np.asarray(raman_tensor)
    if e_in is None:
        I = np.sum(np.abs(R)**2, axis=(-2, -1))
    else:
        e_in = np.asarray(e_in, dtype=complex)
        e_out = e_in if e_out is None else np.asarray(e_out, dtype=complex)
        I = np.abs(np.einsum('...ij,i,j->...', R, e_in, np.conj(e_out)))**2
    if ph_temp is not None:
        ph = np.asarray(ph_energies, dtype=float)
        if ph.ndim == 2: ph = ph[0]
        I = I * (bose(ph, ph_temp) + 1.0)
    return I
