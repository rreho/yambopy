"""cBN: convergence of PL vs number of intermediate Q=0 (Lbar) excitons nexc_in.
Exc-ph computed once for nexc_in=NMAX (in) x 12 (out); PL for each n uses slices.
Window restricted to the phonon-replica region (direct line ~4.7 eV higher is Boltzmann-dead)."""
import json, sys, time, numpy as np
from yambopy.exciton_phonon.excph_input_data import exc_ph_get_inputs
from yambopy.exciton_phonon.excph_luminescence import exc_ph_luminescence, exc_ph_renormalisation
cfg = json.load(open(sys.argv[1])); t0=time.time()
NMAX=cfg["nmax"]; NOUT=cfg["nexc_out"]
ph_E, exc_E, exc_E_in, G, dip = exc_ph_get_inputs(cfg["save"], cfg["elph"], cfg["bse_out"], mode="PL",
    bse_path2=cfg["bse_in"], dipoles_path=cfg["dipoles"], nexc_in=NMAX, nexc_out=NOUT)
print("inputs %.0f s; G"%(time.time()-t0), G.shape, "E_in range %.3f-%.3f eV"%(exc_E_in.min(),exc_E_in.max()), flush=True)
print("lowest Lout %.4f eV; ph max %.1f meV; |dip|^2 first 12:"%(exc_E.min(),ph_E.max()*1e3), np.round(np.sum(np.abs(dip)**2,0)[:12],4), flush=True)
emin=exc_E.min()-ph_E.max()-0.10; emax=exc_E.min()+0.03
out={}
for n in cfg["nlist"]:
    for T in cfg["T"]:
        w,I = exc_ph_luminescence(T, ph_E, exc_E, dip[:,:n], G[:,:,:n,:], exc_energies_in=exc_E_in[:n],
                                  exc_temp=T, nexc_out=NOUT, nexc_in=n, emin=emin, emax=emax,
                                  estep=cfg["estep"], broad=cfg["broad"])
        R = exc_ph_renormalisation(T, ph_E, exc_E, G[:,:,:n,:], exc_energies_in=exc_E_in[:n], nexc_out=NOUT, nexc_in=n)
        out[f"PL_n{n}_T{T}"]=np.real(I); out[f"R_n{n}_T{T}"]=np.real(R)
        k=np.argmax(I); print(f"n={n} T={T}: max at {w[k]:.4f} eV, integral {np.trapz(I,w):.5e}  ({time.time()-t0:.0f} s)", flush=True)
np.savez(cfg["out"], omega=w, nlist=np.array(cfg["nlist"]), T=np.array(cfg["T"],float), exc_energies=exc_E,
         exc_energies_in=exc_E_in, ph_energies=ph_E, broad=cfg["broad"], **out)
print("saved", cfg["out"])
