"""cBN branch-resolved phonon-assisted PL (TA/LA/TO/LO), nexc_in in NLIST, T in TS, broadenings 5 and 25 meV.
Reimplements exc_ph_luminescence (PT prefactor, both channels, broad_0=broad, Lorentzian HWHM=broad/2) with
the satellite weights T[q,m,o] precomputed, so branch masks and broadenings are cheap. Validated against
the library total (n=200, T=10 K, 5 meV) stored in cBN_PL_nexc400_conv_16k.npz.
Branch assignment per q (eigenvectors from ndb.elph): longitudinal fraction L_m = sum_a |qhat.e_ma|^2/sum_a|e_ma|^2;
the 2 modes with largest L are {LA,LO} (lower energy = LA), the 4 others {TA,TA,TO,TO} (2 lowest = TA).
At q=0 (qhat undefined) all modes are labelled by energy (acoustic zeroed by the exc-ph code anyway)."""
import json, numpy as np
from yambopy import YamboLatticeDB, LetzElphElectronPhononDB
from yambopy.units import ha2ev
from yambopy.tools.funcs import bose, boltzman_f
from yambopy.exciton_phonon.excph_input_data import exc_ph_get_inputs
cfg=json.load(open("config.json")); NL=[600]; TS=[10,300]; BR=[0.005,0.025]
ph_E, exc_E, exc_E_in, G, dip = exc_ph_get_inputs(cfg["save"], cfg["elph"], cfg["bse_out"], mode="PL",
    bse_path2=cfg["bse_in"], dipoles_path=cfg["dipoles"], nexc_in=cfg["nmax"], nexc_out=cfg["nexc_out"])
# ---- branch labels
el=LetzElphElectronPhononDB(cfg["elph"]+"/ndb.elph",read_all=False)
import netCDF4
with netCDF4.Dataset(cfg["elph"]+"/ndb.elph") as ds: el.read_eigenmodes(ds)
ev=np.asarray(el.ph_eigenvectors)   # [nq,nm,nat,3]
lat=YamboLatticeDB.from_db_file(filename=cfg["save"]+"/ns.db1")
qc=np.asarray(el.qpoints)@lat.rlat; nq=len(qc); nm=ph_E.shape[1]
lab=np.zeros((nq,nm),int)  # 0 TA 1 LA 2 TO 3 LO
for iq in range(nq):
    e=ev[iq].reshape(nm,-1,3); order=np.argsort(ph_E[iq])
    if np.linalg.norm(qc[iq])<1e-8:
        lab[iq,order]=[0,0,1,2,2,3]; continue
    qh=qc[iq]/np.linalg.norm(qc[iq])
    L=np.sum(np.abs(np.einsum("mai,i->ma",e,qh))**2,axis=1)/np.sum(np.abs(e)**2,axis=(1,2))
    lon=np.argsort(L)[-2:]; tr=np.setdiff1d(np.arange(nm),lon)
    lon=lon[np.argsort(ph_E[iq,lon])]; tr=tr[np.argsort(ph_E[iq,tr])]
    lab[iq,lon[0]]=1; lab[iq,lon[1]]=3; lab[iq,tr[:2]]=0; lab[iq,tr[2:]]=2
iX=np.argmin(exc_E[:,0]); print("q of min exciton", el.qpoints[iX], "phonons meV", np.round(ph_E[iX]*1e3,1), "labels", lab[iX])
z=np.load(cfg["out"]); w=z["omega"].astype(float); W=w/ha2ev
Gc=np.conj(G); Eo=exc_E[:,:cfg["nexc_out"]]; Nq=nq
names=["TA","LA","TO","LO"]; out={}
for n in NL:
    D=dip[:,:n]; Ei=exc_E_in[:n]
    for T in TS:
        occX=boltzman_f(Eo-Eo.min(),T); n_ph=bose(ph_E,T)
        for b in BR:
            spec=np.zeros((4,len(w)))
            for s in (-1,+1):
                se=(Eo[:,None,None,:]-Ei[None,None,:,None]+s*ph_E[:,:,None,None]+1j*b)/ha2ev
                Tq=np.einsum("ri,qmio->rqmo",D,Gc[:,:,:n,:]/se,optimize=True)
                Tq=np.sum(np.abs(Tq)**2,axis=0)/3.*occX[:,None,:]*(n_ph-(s-1)/2.)[:,:,None]
                pole=(Eo[:,None,:]+s*ph_E[:,:,None])/ha2ev; lf=1./pole; h=b/2/ha2ev
                for k in range(4):
                    msk=(lab==k)[:,:,None]; wt=(Tq*lf*msk).ravel(); p=pole.ravel(); sel=wt!=0
                    spec[k]+=np.sum(wt[sel][None,:]*h/np.pi/((W[:,None]-p[sel][None,:])**2+h**2),axis=1)/Nq*ha2ev**2
            tot=spec.sum(0); key=f"n{n}_T{T}_b{int(b*1e3)}"
            out["total_"+key]=tot
            for k in range(4): out[names[k]+"_"+key]=spec[k]
            wts=[np.trapz(spec[k],w) for k in range(4)]
            print(key, "integral", "%.4e"%np.trapz(tot,w), "branch fractions", dict(zip(names,np.round(np.array(wts)/sum(wts),3))),
                  "peak", "%.4f"%w[np.argmax(tot)], flush=True)
            if n==600 and T==10 and b==0.005:
                ref=z["PL_n600_T10"]; print("   validation vs library: max rel diff %.2e"%(np.max(np.abs(tot-ref))/ref.max()))
np.savez("data/cBN_PL_branches600_16k.npz",omega=w,T=np.array(TS,float),nexc_in=np.array(NL),broad=np.array(BR),
         branch_names=np.array(names),exc_energies=exc_E,ph_energies=ph_E,branch_labels=lab,**out)
print("saved cBN_PL_branches600_16k.npz")
