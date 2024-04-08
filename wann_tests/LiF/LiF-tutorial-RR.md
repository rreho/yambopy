# LiF tutorial - From ground state DFT to excited state MBPT via Wannerization methods
In this tutorial, we will compute the electronic groud state properties of LiF via Density Functional Theory (DFT) [^1] using the Quantum Espresso code package [^6]. Afterwards, we will exploit a Wannerization procedure [^2] to compute the electronic band structure *via* Fourier interpolation, increasing the sampling grid at a reduced computational cost. 
Pre- and post- processing of the data are done with [yambopy](https://github.com/rreho/yambopy/tree/devel-tbwannier). In particular, the branch **devel-tbwannier** contains useful routines for IO and interpolation of the electronic (and excitonic) Hamiltonian.

After the ground state properties are assessed, we will analyze the excited state with Yambo[^4][^5]. In particular, we will compute the absorption spectra and the exciton band structure.
We apply a Wannierization procedure for the excitonic Hamiltonian $H^{2p}$ similar to the electronic case and compare our results with the literature [^3]. 

All simulation employ **LDA** pseudopotentials without Spin-Orbit coupling (SOC).
## Crystal structure LiF
LiF has a face centered cubic (fcc) crystal structure with 48 symmetry operations.
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/BJrSA7ZxA.png" alt="LiF" width="70%">
</div>

The high-symmetry band structure we employ is : $W-L-\Gamma-X-W$
## DFT
### self-consistent field calculation
The first step consist in computing the electronic charge density of the auxiliary Kohn-Sham system via a self-consistent field calculation.
1. Create an input file ```scf.in```:
```bash
  &control
    calculation = 'scf',
    verbosity='high'
    pseudo_dir   = "../psps/",
    outdir       = ".",
    wf_collect=.TRUE.,
    prefix='LiF',
 /
 &system
    ibrav = 2, nat = 2, ntyp = 2,
    force_symmorphic=.TRUE. ,
    ecutwfc = 80.0,
    nbnd = 10 ,
    celldm(1)=7.703476,
 /
 &electrons
    diago_full_acc = .TRUE.,
    conv_thr =  1.0d-10
 /
 ATOMIC_SPECIES
    Li    6.941       Li.LDA.cpi.UPF
     F    18.998403   F.LDA.cpi.UPF
 ATOMIC_POSITIONS crystal
Li  0. 0. 0.
 F  0.5 -0.5 -0.5
K_POINTS automatic
 4  4  4  0  0  0
```
2. run with:
```
mpirun -np 4 pw.x < scf.in |tee log_scf.out
```
### non-self-consistent calculation
We build the database for Yambo via a **non-self-consistent calculation (nscf)**.
1. Create and go in a new directory `mkdir nscf`
2. Copy the `.save` folder from the previous `scf` run `cp -r ../scf/LiF.save .` 
3. Create an `nscf.in` input file:
```bash
 &control
    calculation = 'nscf',
    verbosity='high'
    pseudo_dir   = "../psps"
    outdir       = "."
    wf_collect=.TRUE.,
    prefix='LiF',
 /
 &system
    ibrav = 2, nat = 2, ntyp = 2,
    force_symmorphic=.TRUE. ,
    ecutwfc = 80.0,
    nbnd = 50 ,
    celldm(1)=7.703476,
 /
 &electrons
    diago_full_acc = .TRUE.,
    conv_thr =  1.0d-10
 /
 ATOMIC_SPECIES
    Li    6.941       Li.LDA.cpi.UPF
     F    18.998403   F.LDA.cpi.UPF
 ATOMIC_POSITIONS crystal
Li  0. 0. 0.
 F  0.5 -0.5 -0.5
K_POINTS automatic
 4  4  4  0  0  0
```
4. run:
```
mpirun -np 4 pw.x < nscf.in |tee log_nscf.out
```
### Band structure
We can get insight into the electronic states of the system computing the band structure along the high-symmetry Brillouin Zone (BZ) path $W-L-\Gamma-X-W$.
In addition, we compute the wavefunctions projected to atomic orbitals via `projwfc.x` in order to inspect the orbital projected band structure.
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/r1EIqE-xC.png" alt="bz-path" width=40% pos=center>
</div> 
1. Create and move to a new `bands` folder
2. Copy the `scf` `.save` folder in the current directory `cp -r ../scf/LiF.save .`
3. create a new input file `nscf.in` (in QE a `bands` and `nscf` calculation are equivalent)
```bash 
  &control
    calculation = 'nscf',
    verbosity='high'
    pseudo_dir   = "../psps"
    outdir       = "."
    wf_collect=.TRUE.,
    prefix='LiF',
 /
 &system
    ibrav = 2, nat = 2, ntyp = 2,
    force_symmorphic=.TRUE. ,
    ecutwfc = 80.0,
    nbnd = 10 ,
    celldm(1)=7.703476,
 /
 &electrons
    diago_full_acc = .TRUE.,
    conv_thr =  1.0d-10
 /
 ATOMIC_SPECIES
    Li    6.941       Li.LDA.cpi.UPF
     F    18.998403   F.LDA.cpi.UPF
 ATOMIC_POSITIONS crystal
Li  0. 0. 0.
 F  0.5 -0.5 -0.5
K_POINTS crystal_b
121
0.5    0.25    0.75    1
0.5    0.25833333333333336    0.7416666666666667    1
0.5    0.26666666666666666    0.7333333333333333    1
0.5    0.275    0.725    1
0.5    0.2833333333333333    0.7166666666666667    1
0.5    0.2916666666666667    0.7083333333333334    1
0.5    0.3    0.7    1
0.5    0.30833333333333335    0.6916666666666667    1
0.5    0.31666666666666665    0.6833333333333333    1
0.5    0.325    0.675    1
0.5    0.3333333333333333    0.6666666666666666    1
0.5    0.3416666666666667    0.6583333333333333    1
0.5    0.35    0.65    1
0.5    0.35833333333333334    0.6416666666666666    1
0.5    0.3666666666666667    0.6333333333333333    1
0.5    0.375    0.625    1
0.5    0.3833333333333333    0.6166666666666667    1
0.5    0.39166666666666666    0.6083333333333334    1
0.5    0.4    0.6    1
0.5    0.4083333333333333    0.5916666666666667    1
0.5    0.41666666666666663    0.5833333333333334    1
0.5    0.425    0.575    1
0.5    0.43333333333333335    0.5666666666666667    1
0.5    0.44166666666666665    0.5583333333333333    1
0.5    0.45    0.55    1
0.5    0.45833333333333337    0.5416666666666666    1
0.5    0.4666666666666667    0.5333333333333333    1
0.5    0.475    0.525    1
0.5    0.48333333333333334    0.5166666666666666    1
0.5    0.4916666666666667    0.5083333333333333    1
0.5    0.5    0.5    1
0.48333333333333334    0.48333333333333334    0.48333333333333334    1
0.4666666666666667    0.4666666666666667    0.4666666666666667    1
0.45    0.45    0.45    1
0.43333333333333335    0.43333333333333335    0.43333333333333335    1
0.4166666666666667    0.4166666666666667    0.4166666666666667    1
0.4    0.4    0.4    1
0.3833333333333333    0.3833333333333333    0.3833333333333333    1
0.3666666666666667    0.3666666666666667    0.3666666666666667    1
0.35    0.35    0.35    1
0.33333333333333337    0.33333333333333337    0.33333333333333337    1
0.31666666666666665    0.31666666666666665    0.31666666666666665    1
0.3    0.3    0.3    1
0.2833333333333333    0.2833333333333333    0.2833333333333333    1
0.26666666666666666    0.26666666666666666    0.26666666666666666    1
0.25    0.25    0.25    1
0.23333333333333334    0.23333333333333334    0.23333333333333334    1
0.21666666666666667    0.21666666666666667    0.21666666666666667    1
0.2    0.2    0.2    1
0.18333333333333335    0.18333333333333335    0.18333333333333335    1
0.16666666666666669    0.16666666666666669    0.16666666666666669    1
0.15000000000000002    0.15000000000000002    0.15000000000000002    1
0.13333333333333336    0.13333333333333336    0.13333333333333336    1
0.11666666666666664    0.11666666666666664    0.11666666666666664    1
0.09999999999999998    0.09999999999999998    0.09999999999999998    1
0.08333333333333331    0.08333333333333331    0.08333333333333331    1
0.06666666666666665    0.06666666666666665    0.06666666666666665    1
0.04999999999999999    0.04999999999999999    0.04999999999999999    1
0.033333333333333326    0.033333333333333326    0.033333333333333326    1
0.016666666666666663    0.016666666666666663    0.016666666666666663    1
0.0    0.0    0.0    1
0.016666666666666666    0.0    0.016666666666666666    1
0.03333333333333333    0.0    0.03333333333333333    1
0.05    0.0    0.05    1
0.06666666666666667    0.0    0.06666666666666667    1
0.08333333333333333    0.0    0.08333333333333333    1
0.1    0.0    0.1    1
0.11666666666666667    0.0    0.11666666666666667    1
0.13333333333333333    0.0    0.13333333333333333    1
0.15    0.0    0.15    1
0.16666666666666666    0.0    0.16666666666666666    1
0.18333333333333332    0.0    0.18333333333333332    1
0.2    0.0    0.2    1
0.21666666666666667    0.0    0.21666666666666667    1
0.23333333333333334    0.0    0.23333333333333334    1
0.25    0.0    0.25    1
0.26666666666666666    0.0    0.26666666666666666    1
0.2833333333333333    0.0    0.2833333333333333    1
0.3    0.0    0.3    1
0.31666666666666665    0.0    0.31666666666666665    1
0.3333333333333333    0.0    0.3333333333333333    1
0.35    0.0    0.35    1
0.36666666666666664    0.0    0.36666666666666664    1
0.38333333333333336    0.0    0.38333333333333336    1
0.4    0.0    0.4    1
0.4166666666666667    0.0    0.4166666666666667    1
0.43333333333333335    0.0    0.43333333333333335    1
0.45    0.0    0.45    1
0.4666666666666667    0.0    0.4666666666666667    1
0.48333333333333334    0.0    0.48333333333333334    1
0.5    0.0    0.5    1
0.5    0.008333333333333333    0.5083333333333333    1
0.5    0.016666666666666666    0.5166666666666667    1
0.5    0.025    0.525    1
0.5    0.03333333333333333    0.5333333333333333    1
0.5    0.041666666666666664    0.5416666666666666    1
0.5    0.05    0.55    1
0.5    0.058333333333333334    0.5583333333333333    1
0.5    0.06666666666666667    0.5666666666666667    1
0.5    0.075    0.575    1
0.5    0.08333333333333333    0.5833333333333334    1
0.5    0.09166666666666666    0.5916666666666667    1
0.5    0.1    0.6    1
0.5    0.10833333333333334    0.6083333333333334    1
0.5    0.11666666666666667    0.6166666666666667    1
0.5    0.125    0.625    1
0.5    0.13333333333333333    0.6333333333333333    1
0.5    0.14166666666666666    0.6416666666666666    1
0.5    0.15    0.65    1
0.5    0.15833333333333333    0.6583333333333333    1
0.5    0.16666666666666666    0.6666666666666666    1
0.5    0.175    0.675    1
0.5    0.18333333333333332    0.6833333333333333    1
0.5    0.19166666666666668    0.6916666666666667    1
0.5    0.2    0.7    1
0.5    0.20833333333333334    0.7083333333333334    1
0.5    0.21666666666666667    0.7166666666666667    1
0.5    0.225    0.725    1
0.5    0.23333333333333334    0.7333333333333334    1
0.5    0.24166666666666667    0.7416666666666667    1
0.5    0.25    0.75    1
```
Note that we created a list of k-points along the high-symmetry path using **yambopy**. After importing the correct libraries you can create an instance of the `Path` class along the given path with `npoints` between each high-symmetry point. 
```python=
# Define path in reduced coordinates using Class Path
npoints = 30
path_kpoints = Path([[[  0.5,  0.250,  0.750],'W'],
                     [[0.5,0.5,  0.5],'L'],  
              [[  0.0,  0.0,  0.0],'$\Gamma$'],
              [[  0.5,  0.0,  0.5],'X'],
              [[  0.5,  0.250,  0.750],'W']],[npoints,npoints,npoints,npoints] )
```
2. run: 
``` mpirun -np 4 pw.x < nscf.in |tee ```
3. Compute the orbital projected wavefunction. Create a new `bands.in` file
```bash
&projwfc
    prefix = 'LiF',
    outdir = './',
    ngauss=0, degauss=0.036748
    kresolveddos=.true. ! optional
    DeltaE=0.01
    filpdos = 'LiF-pdos.dat'
 /
```
an run 
```
mpirun -np 4 projwfc.x < bands.in |tee projwfc.log
```
**Note** the output file has to be named `projwfc.log` for yambopy

4. Inspect the output file `projwfc.log` to get the indices of the state belonging to a given atom's orbital
```
     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Li ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Li ), wfc  2 (l=1 m= 1)
...
```
5. Plot the orbital projected band structure via `yambopy`:
```python=
# import libraries
from yambopy import *
import matplotlib.pyplot as plt
from yambopy.lattice import car_red, red_car
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines

# create a figure and ax
fig,ax = plt.subplots(dpi=300)

dotsize = 10 #size of the dots, change at will
# Read the indices from `projwfc.log`. Remember python counting starts from 0

atom_Li_s = [0]
atom_Li_p = [1,2,3]
atom_Li_d = [4,5,6,7,8]
atom_Li_f = [9,10,11,12,13,14,15,15]
atom_F_s = [16]
atom_F_p = [17,18,19]
atom_F_d = [20,21,22,23,24,25,26]
atom_F_f = [27,28,29,30,31]
# get ticks and labels, need previous istance of Path
ticks, labels =list(zip(*path_kpoints.get_indexes()))

# create an instance of ProjwfcXML class
band = ProjwfcXML(prefix='LiF',path=f'./bands/',qe_version='6.7')
nelectrons = 8
Li_s = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_Li_s,color='pink',size=dotsize)
Li_p = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_Li_p,color='yellow',size=dotsize)
Li_d = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_Li_d,color='orange',size=dotsize)
Li_f = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_Li_f,color='red',size=dotsize)
F_s = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_F_s,color='blue',size=dotsize)
F_p = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_F_p,color='indigo',size=dotsize)
F_d = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_F_d,color='cyan',size=dotsize)
F_f = band.plot_eigen(ax,path_kpoints=path_kpoints,selected_orbitals=atom_F_f,color='green',size=dotsize)

# setting labels and legend of the plot
Li_s.set_label(r'$Li_s$')
Li_p.set_label(r"$Li_p$")
Li_d.set_label(r'$Li_d$')
Li_f.set_label(r"$Li_f$")
F_s.set_label(r'$F_s$')
F_p.set_label(r'$F_p$')
F_d.set_label(r'$F_d$')
F_f.set_label(r'$F_f$')
ax.set_ylabel('E [eV]')
#ax.set_ylim([-5,20])
lgnd = plt.legend(loc=(1.04, 0), scatterpoints=1, markerscale=20, fontsize=15)
lgnd.legendHandles[0]._sizes = [dotsize]
lgnd.legendHandles[1]._sizes = [dotsize]
lgnd.legendHandles[2]._sizes = [dotsize]
lgnd.legendHandles[3]._sizes = [dotsize]
lgnd.legendHandles[4]._sizes = [dotsize]
lgnd.legendHandles[5]._sizes = [dotsize]
lgnd.legendHandles[6]._sizes = [dotsize]
lgnd.legendHandles[7]._sizes = [dotsize]
#save figure
plt.savefig(f'./bands/full_orbbands.pdf',bbox_inches='tight')
```
You can play with the aesthetic and plotting options. The final result should look like this

<img src="https://hackmd.io/_uploads/HJA3REbeR.png" alt="full_orbbands" style="width: 48%; margin-right: 2%;">
<img src="https://hackmd.io/_uploads/ry-8Jr-lA.png" alt="orbbands" style="width: 48%;">


From these two figure we can see that the three upmost valence bands are mainly composed of Fluorine p orbitals while the first conduction band has a mixed $Li_p$, $Li_s$, $F_s$, $F_p$ orbital. This information is important for Wannierization and it slightly differs from the analysis carried in the reference [^3], where they only considered $Li_s$ and $F_p$ orbitals.
### Wannierization
**Requirement** go into the [nscf](#non-self-consistent-calculation) `.save` folder and initialize the yambo database with `p2y` and `yambo`.

In order to run a Wannier90 calculation we first need to run an [nscf](#non-self-consistent-calculation) with a uniform kgrid.
We will consider two grids, a **k-grid = 8x8x8** and a **q-grid=4x4x4**. We report the steps for the q-grid.
1. You can generate a uniform kgrid with the `kmesh.pl` utility provided by the `wannier90` package.
However, in order to be ensure consistency between the k-grid employed by `Yambo` and `QE` we use `yambopy` to print the list of k-points in the `full BZ`
```python=
# 
savedb_k = YamboSaveDB.from_db_file(f'./database/SAVE')
for i in range (len(savedb_q.red_kpoints)):
    print(f'{savedb_q.red_kpoints[i][0]:.10f}   {savedb_q.red_kpoints[i][1]:.10f}   {savedb_q.red_kpoints[i][2]:.10f} {1/len(savedb_q.red_kpoints):.10f}')
```
2. Create an `nscf.in` input file
```bash=
  &control
    calculation = 'nscf',
    verbosity='high'
    pseudo_dir   = "../psps"
    outdir       = "."
    wf_collect=.TRUE.,
    prefix='LiF',
 /
 &system
    ibrav = 2, nat = 2, ntyp = 2,
    force_symmorphic=.TRUE. ,
    ecutwfc = 80.0,
    nbnd = 10 ,
    celldm(1)=7.703476,
 /
 &electrons
    diago_full_acc = .TRUE.,
    conv_thr =  1.0d-10
 /
 ATOMIC_SPECIES
    Li    6.941       Li.LDA.cpi.UPF
     F    18.998403   F.LDA.cpi.UPF
 ATOMIC_POSITIONS crystal
Li  0. 0. 0.
 F  0.5 -0.5 -0.5
K_POINTS crystal
64
  0.00000000  0.00000000  0.00000000  1.562500e-02
  0.00000000  0.00000000  0.25000000  1.562500e-02
  0.00000000  0.00000000  0.50000000  1.562500e-02
  0.00000000  0.00000000  0.75000000  1.562500e-02
  0.00000000  0.25000000  0.00000000  1.562500e-02
  0.00000000  0.25000000  0.25000000  1.562500e-02
  0.00000000  0.25000000  0.50000000  1.562500e-02
  0.00000000  0.25000000  0.75000000  1.562500e-02
  0.00000000  0.50000000  0.00000000  1.562500e-02
  0.00000000  0.50000000  0.25000000  1.562500e-02
  0.00000000  0.50000000  0.50000000  1.562500e-02
  0.00000000  0.50000000  0.75000000  1.562500e-02
  0.00000000  0.75000000  0.00000000  1.562500e-02
  0.00000000  0.75000000  0.25000000  1.562500e-02
  0.00000000  0.75000000  0.50000000  1.562500e-02
  0.00000000  0.75000000  0.75000000  1.562500e-02
  0.25000000  0.00000000  0.00000000  1.562500e-02
  0.25000000  0.00000000  0.25000000  1.562500e-02
  0.25000000  0.00000000  0.50000000  1.562500e-02
  0.25000000  0.00000000  0.75000000  1.562500e-02
  0.25000000  0.25000000  0.00000000  1.562500e-02
  0.25000000  0.25000000  0.25000000  1.562500e-02
  0.25000000  0.25000000  0.50000000  1.562500e-02
  0.25000000  0.25000000  0.75000000  1.562500e-02
  0.25000000  0.50000000  0.00000000  1.562500e-02
  0.25000000  0.50000000  0.25000000  1.562500e-02
  0.25000000  0.50000000  0.50000000  1.562500e-02
  0.25000000  0.50000000  0.75000000  1.562500e-02
  0.25000000  0.75000000  0.00000000  1.562500e-02
  0.25000000  0.75000000  0.25000000  1.562500e-02
  0.25000000  0.75000000  0.50000000  1.562500e-02
  0.25000000  0.75000000  0.75000000  1.562500e-02
  0.50000000  0.00000000  0.00000000  1.562500e-02
  0.50000000  0.00000000  0.25000000  1.562500e-02
  0.50000000  0.00000000  0.50000000  1.562500e-02
  0.50000000  0.00000000  0.75000000  1.562500e-02
  0.50000000  0.25000000  0.00000000  1.562500e-02
  0.50000000  0.25000000  0.25000000  1.562500e-02
  0.50000000  0.25000000  0.50000000  1.562500e-02
  0.50000000  0.25000000  0.75000000  1.562500e-02
  0.50000000  0.50000000  0.00000000  1.562500e-02
  0.50000000  0.50000000  0.25000000  1.562500e-02
  0.50000000  0.50000000  0.50000000  1.562500e-02
  0.50000000  0.50000000  0.75000000  1.562500e-02
  0.50000000  0.75000000  0.00000000  1.562500e-02
  0.50000000  0.75000000  0.25000000  1.562500e-02
  0.50000000  0.75000000  0.50000000  1.562500e-02
  0.50000000  0.75000000  0.75000000  1.562500e-02
  0.75000000  0.00000000  0.00000000  1.562500e-02
  0.75000000  0.00000000  0.25000000  1.562500e-02
  0.75000000  0.00000000  0.50000000  1.562500e-02
  0.75000000  0.00000000  0.75000000  1.562500e-02
  0.75000000  0.25000000  0.00000000  1.562500e-02
  0.75000000  0.25000000  0.25000000  1.562500e-02
  0.75000000  0.25000000  0.50000000  1.562500e-02
  0.75000000  0.25000000  0.75000000  1.562500e-02
  0.75000000  0.50000000  0.00000000  1.562500e-02
  0.75000000  0.50000000  0.25000000  1.562500e-02
  0.75000000  0.50000000  0.50000000  1.562500e-02
  0.75000000  0.50000000  0.75000000  1.562500e-02
  0.75000000  0.75000000  0.00000000  1.562500e-02
  0.75000000  0.75000000  0.25000000  1.562500e-02
  0.75000000  0.75000000  0.50000000  1.562500e-02
  0.75000000  0.75000000  0.75000000  1.562500e-02
```
3. Create an input file for wannier90 `LiF.win`:
```bash=
num_bands         =   10
num_wann          =   8


DIS_WIN_MIN = -25
DIS_WIN_MAX = 24
dis_num_iter      = 1000
dis_mix_ratio     = 0.4

num_iter          = 3000
iprint    = 3
!num_dump_cycles = 10
!num_print_cycles = 1000


bands_num_points = 201
bands_plot_format = gnuplot

write_rmn = true
write_tb  = true
write_xyz = .true.
wannier_plot_supercell = 3
use_ws_distance = .true.
translate_home_cell = .true.

bands_plot = true
write_hr = true
Fermi_energy = 3.63060
Begin Atoms_Frac
F            0.5000000000       -0.5000000000      -0.5000000000
Li           0.0000000000       0.0000000000       0.0000000000
End Atoms_Frac

Begin Projections
Li :l=0;l=1
F : l=0;l=1
End Projections

Begin kpoint_path
W  0.500    0.250    0.750 L  0.500    0.500    0.500
L  0.500    0.500    0.500 Γ  0.000    0.000    0.00
Γ  0.000    0.000    0.00 X  0.500    0.000    0.500
X  0.500    0.000    0.500 W  0.500    0.250    0.750
End kpoint_path

Begin Unit_Cell_Cart
Bohr
     -3.851738  0.0000000000      3.851738
     0.000000000000000   3.851738 3.851738
     -3.851738  3.851738 0.000000000000000
End Unit_Cell_Cart

mp_grid      = 4 4 4

!exclude_bands 1,6,7,8,9,10
Begin kpoints
  0.00000000  0.00000000  0.00000000  1.562500e-02
  0.00000000  0.00000000  0.25000000  1.562500e-02
  0.00000000  0.00000000  0.50000000  1.562500e-02
  0.00000000  0.00000000  0.75000000  1.562500e-02
  0.00000000  0.25000000  0.00000000  1.562500e-02
  0.00000000  0.25000000  0.25000000  1.562500e-02
  0.00000000  0.25000000  0.50000000  1.562500e-02
  0.00000000  0.25000000  0.75000000  1.562500e-02
  0.00000000  0.50000000  0.00000000  1.562500e-02
  0.00000000  0.50000000  0.25000000  1.562500e-02
  0.00000000  0.50000000  0.50000000  1.562500e-02
  0.00000000  0.50000000  0.75000000  1.562500e-02
  0.00000000  0.75000000  0.00000000  1.562500e-02
  0.00000000  0.75000000  0.25000000  1.562500e-02
  0.00000000  0.75000000  0.50000000  1.562500e-02
  0.00000000  0.75000000  0.75000000  1.562500e-02
  0.25000000  0.00000000  0.00000000  1.562500e-02
  0.25000000  0.00000000  0.25000000  1.562500e-02
  0.25000000  0.00000000  0.50000000  1.562500e-02
  0.25000000  0.00000000  0.75000000  1.562500e-02
  0.25000000  0.25000000  0.00000000  1.562500e-02
  0.25000000  0.25000000  0.25000000  1.562500e-02
  0.25000000  0.25000000  0.50000000  1.562500e-02
  0.25000000  0.25000000  0.75000000  1.562500e-02
  0.25000000  0.50000000  0.00000000  1.562500e-02
  0.25000000  0.50000000  0.25000000  1.562500e-02
  0.25000000  0.50000000  0.50000000  1.562500e-02
  0.25000000  0.50000000  0.75000000  1.562500e-02
  0.25000000  0.75000000  0.00000000  1.562500e-02
  0.25000000  0.75000000  0.25000000  1.562500e-02
  0.25000000  0.75000000  0.50000000  1.562500e-02
  0.25000000  0.75000000  0.75000000  1.562500e-02
  0.50000000  0.00000000  0.00000000  1.562500e-02
  0.50000000  0.00000000  0.25000000  1.562500e-02
  0.50000000  0.00000000  0.50000000  1.562500e-02
  0.50000000  0.00000000  0.75000000  1.562500e-02
  0.50000000  0.25000000  0.00000000  1.562500e-02
  0.50000000  0.25000000  0.25000000  1.562500e-02
  0.50000000  0.25000000  0.50000000  1.562500e-02
  0.50000000  0.25000000  0.75000000  1.562500e-02
  0.50000000  0.50000000  0.00000000  1.562500e-02
  0.50000000  0.50000000  0.25000000  1.562500e-02
  0.50000000  0.50000000  0.50000000  1.562500e-02
  0.50000000  0.50000000  0.75000000  1.562500e-02
  0.50000000  0.75000000  0.00000000  1.562500e-02
  0.50000000  0.75000000  0.25000000  1.562500e-02
  0.50000000  0.75000000  0.50000000  1.562500e-02
  0.50000000  0.75000000  0.75000000  1.562500e-02
  0.75000000  0.00000000  0.00000000  1.562500e-02
  0.75000000  0.00000000  0.25000000  1.562500e-02
  0.75000000  0.00000000  0.50000000  1.562500e-02
  0.75000000  0.00000000  0.75000000  1.562500e-02
  0.75000000  0.25000000  0.00000000  1.562500e-02
  0.75000000  0.25000000  0.25000000  1.562500e-02
  0.75000000  0.25000000  0.50000000  1.562500e-02
  0.75000000  0.25000000  0.75000000  1.562500e-02
  0.75000000  0.50000000  0.00000000  1.562500e-02
  0.75000000  0.50000000  0.25000000  1.562500e-02
  0.75000000  0.50000000  0.50000000  1.562500e-02
  0.75000000  0.50000000  0.75000000  1.562500e-02
  0.75000000  0.75000000  0.00000000  1.562500e-02
  0.75000000  0.75000000  0.25000000  1.562500e-02
  0.75000000  0.75000000  0.50000000  1.562500e-02
  0.75000000  0.75000000  0.75000000  1.562500e-02
end kpoints
```
For the initial projections see the discussion in [Band structure](#Band-structure)

3. Copy the `scf` `.save` folder `cp -r ../scf/LiF.save .` (for the k-grid you might want to run another scf run with an 8x8x8 grid to ensure consistency)
4. Run an `nscf` calculation
```
mpirun -np 4 pw.x < nscf.in |tee log_nscf.out
```
5. Run the pre-processing steps of Wannier90
```
mpirun -np 1 wannier90.x -pp LiF
```
6. Create an input file `pw2wan.in` for `pw2wannier90.x` and run `pw2wannier90.x`:
```bash 
&inputpp
outdir = './'
prefix = 'LiF'
seedname = 'LiF'
write_amn = .true.
write_mmn = .true.
write_unk = .true. !optional
/
```
```
mpirun -np 1 pw2wannier90.x < pw2wan.in |tee log_pw2wan.out
```
7. Finally run `wannier90.x`
```
mpirun -np 1 wannier90.x LiF
```
You should now have a `seedname_hr.dat` file containing the real-space Hamiltonian in the MLWF basis.
We can compare the electronic band structure obtained with `Wannier90` and `QE` via the following Python scripting.
```python=
import tbmodels
from yambopy import *
import matplotlib.pyplot as plt
from yambopy.lattice import car_red, red_car
import matplotlib.pylab as pylab
from pylab import rcParams
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines

WORK_PATH='./'
unshifted_gridpath='yambo_tutorials/unshifted-grid/'
# Read xml file from a bands calculation
xml = ProjwfcXML(prefix='LiF',path=f'{unshiftedgrid_path}/bands/',qe_version='6.7')

# Compare DFT vs Wannier band structure
from qepy.lattice import Path, calculate_distances 
# Class PwXML. QE database reading
xml = ProjwfcXML(prefix='LiF',path=f'{unshiftedgrid_path}/bands/',qe_version='6.7')
wann_bands = np.loadtxt(f'{unshiftedgrid_path}/nscf-wannier-qgrid/LiF_band.dat',usecols=(0,1))
# Class PwXML. QE database reading
fig, ax = plt.subplots()
kpoints_dists = calculate_distances(xml.kpoints[:xml.nkpoints])
#xml.plot_eigen_ax(ax, path_kpoints, y_offset=-1., lw =1, ylim=(-4,17))
for ib in range(xml.nbands):
  ax.plot(kpoints_dists, xml.eigen[:,ib], c='red',lw=1.0)
ax.scatter(wann_bands[:,0]/np.max(wann_bands[:,0])*np.max(kpoints_dists), wann_bands[:,1], c='black',s=0.5)
# tb_eb = tb_ebands.add_kpath_labels(ax)
legend_entries = [
    mlines.Line2D([], [], color='black', ls ='-', label='Wannier'),
    mlines.Line2D([], [], color='red', label='DFT'),
]

# # Add custom legend outside the loop
ax.legend(handles=legend_entries, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3)
# ax.plot(qe_en, qe_bands)
ax.set_ylabel('E [eV]')
ax.set_ylim(-5,26)
#plt.show()
plt.savefig(f'{unshiftedgrid_path}/bands/DFTvsWann_q.pdf',bbox_inches='tight')

```
We can now compare the results obtained Wannierizing the k- and q-grid with the DFT computed band structure

<img src="https://hackmd.io/_uploads/BJlrTvZxC.png" alt="dftvswann_k" style="width: 48%; margin-right: 2%;">
<img src="https://hackmd.io/_uploads/ryiSaw-x0.png" alt="dftvswann_q" style="width: 48%;">
The two plots are reasonable in agreement but not ideal. We recover the degeneracy at W in the valence bands manifold. 
It is possible to increase the accurarcy of Wannier interpolation increasing the zie of the grid. Howver, I am not sure this is probelm for this case.

# Excited state - YAMBO
**Requirements**: Run the DFT [scf and nscf](#DFT) calculations. You can download all input and references file [here](https://media.yambo-code.eu/educational/tutorials/files/LiF.tar.gz)

We can move now the the excited state properties computed via *ab initio* Many Body Perturbation Theory (Ai-MBPT) code by Yambo.
As opposite to what has been done [above](#DFT), we will work with *shifted* k-grids 
```
...
K_POINTS automatic
 4  4  4  1  1  1
```
and compare the results obtained with *shifted* and *unshifted* grids.
If you successfully followed the [previous steps][#DFT], you should have a `LiF.save` directory:
```
ls LiF.save
charge-density.dat  data-file-schema.xml  F.LDA.cpi.UPF  Li.LDA.cpi.UPF  wfc10.dat  wfc1.dat  wfc2.dat	wfc3.dat  wfc4.dat  wfc5.dat  wfc6.dat	wfc7.dat 
```
## Conversion to Yambo format
The PWscf `LiF.save` outpu is converted to the Yambo format using the `p2y` executable, found in the yambo `bin` directory. Enter `LiF.save` and launch `p2y`:
```
$ cd LiF.save
$ p2y
...
 <--->  XC potential           : Slater exchange(X)+Perdew & Zunger(C)
 <--->  Atomic species        :  2
 <--->  Max atoms/species     :  1
 <---> == DB1 (Gvecs and more) ...
 <---> ... Database done
 <---> == DB2 (wavefunctions)  ...
 <---> [p2y] WF I/O |                                        | [000%] --(E) --(X)
 <---> [p2y] WF I/O |########################################| [100%] --(E) --(X)
 <---> == DB3 (PseudoPotential) ...
 <--->  == P2Y completed ==
```
This output repeats some information about the system and generates a `SAVE` directory that contains among others the `ns.db1` database with information about the crystal structure of the system.
## Initialization
The initialization procedure is common to all yambo runs.
1. Enter the `YAMBO` directory (where you have previously created the `SAVE` folder) and type
```bash
$ yambo -F Inputs/01_init -J 01_init
```
and you will see:
```
 <---> [03.01] X indexes
 <---> X [eval] |                                        | [000%] --(E) --(X)
 <---> X [eval] |########################################| [100%] --(E) --(X)
 <---> X[REDUX] |                                        | [000%] --(E) --(X)
 <---> X[REDUX] |########################################| [100%] --(E) --(X)
 <---> [03.01.01] Sigma indexes
 <---> Sigma [eval] |                                        | [000%] --(E) --(X)
 <---> Sigma [eval] |########################################| [100%] --(E) --(X)
 <---> Sigma[REDUX] |                                        | [000%] --(E) --(X)
 <---> Sigma[REDUX] |########################################| [100%] --(E) --(X)
 <---> [04] Timing Overview
 <---> [05] Memory Overview
 <---> [06] Game Over & Game summary
```
this run produced an `r_setup` file with a lot of useful information about your system, like the Fermi level, the crystal structure, the number of k-points, the gap etc...

## Random-Phase approximation
Here, we compute the absorption spectrum for LiF with different level of approximations for the kernel.
The simplest approximation that can be used to calculate the absorption spectrum of LiF is the independent particle approximation (`02_RPA_no_LF`).
We can then include a kernel in the Hartree apporximation (`03_RPA_LF`), which correspond to the Random-Phase-Approximation (RPA).
In addition, we can rigidly shift with a scissor the electronic states to match the experimental absorption peak (`03_RPA_LF_QP`)
1. Create an input file called `02_RPA_no_LF`
```
optics                       # [R OPT] Optics
chi                          # [R LR] Linear Response.
% QpntsRXd
  1 | 1 |                   # [Xd] Transferred momenta
%
% BndsRnXd
   1 |  10 |                 # [Xd] Polarization function bands
%
NGsBlkXd= 1              RL  # [Xd] Response block size
% EnRngeXd
  7.50000 | 25.00000 |   eV  # [Xd] Energy range
%
% DmRngeXd
  0.10000 |  0.30000 |   eV  # [Xd] Damping range
%
ETStpsXd= 300                # [Xd] Total Energy steps
% LongDrXd
 1.000000 | 0.000000 | 0.000000 |      # [Xd] [cc] Electric Field
%
```
2. Run yambo:
```
$ yambo -F 02_RPA_no_LF -J 02_RPA_no_LF
```
The optional *-J* flag is used to label the output/report/log files.

3. Create the input files `03_RPA_LF` and `03_RPA_LF_QP`:
```
optics                       # [R OPT] Optics
chi                          # [R LR] Linear Response.
% QpntsRXd
  1 | 1 |                   # [Xd] Transferred momenta
%
% BndsRnXd
   1 |  10 |                 # [Xd] Polarization function bands
%
NGsBlkXd=51              RL  # [Xd] Response block size
% EnRngeXd
  7.50000 | 25.00000 |   eV  # [Xd] Energy range
%
% DmRngeXd
  0.10000 |  0.30000 |   eV  # [Xd] Damping range
%
ETStpsXd= 300                # [Xd] Total Energy steps
% LongDrXd
 1.000000 | 0.000000 | 0.000000 |      # [Xd] [cc] Electric Field
%
```
<span style="color: red;">
% XfnQP_E 
    
 5.190000 | 1.000000 | 1.000000 |      # [EXTQP Xd] E parameters (c/v), scissor
    
%
    
</span> 

and run yambo
```
$ yambo -F 03_RPA_LF -J -03_RPA_LF
$ yambo -F 03_RPA_LF -J -03_RPA_LF_QP
```

4. Plot the output data via
```python 
# read files
data_02_RPA_no_LF = np.loadtxt(f'{YAMBO_TUT_PATH}/shifted-grid/o-02_RPA_no_LF.eps_q1_inv_rpa_dyson', usecols=(0,1))
data_03_RPA_LF = np.loadtxt(f'{YAMBO_TUT_PATH}/shifted-grid/o-03_RPA_LF.eps_q1_inv_rpa_dyson', usecols=(0,1))
data_03_RPA_LF_QP = np.loadtxt(f'{YAMBO_TUT_PATH}/shifted-grid/o-03_RPA_LF_QP.eps_q1_inv_rpa_dyson', usecols=(0,1))
data_exp = np.loadtxt(f'{YAMBO_TUT_PATH}/e2_experimental.dat', usecols=(0,1))
#plot
fig,ax =plt.subplots()
ax.plot(data_02_RPA_no_LF[:,0], data_02_RPA_no_LF[:,1], label='02_RPA_no_LF')
ax.plot(data_03_RPA_LF[:,0], data_03_RPA_LF[:,1], label='03_RPA_LF')
ax.plot(data_03_RPA_LF_QP[:,0], data_03_RPA_LF_QP[:,1], label='03_RPA_LF_QP')
ax.plot(data_exp[:,0], data_exp[:,1], label='Experiment')
ax.set_ylabel('Absorption [a.u.]')
ax.set_xlabel('Energy [eV]')
ax.legend()
plt.savefig(f'{YAMBO_TUT_PATH}/shifted-grid/absorption_tut_v1.png', bbox_inches='tight')
```

![absorption_tut_v1](https://hackmd.io/_uploads/SJnNyFZl0.png)
 
As we can see, the energy shifting helps with matching the experimental energy onset but there is an experimental peak below the fundamental peak which is not captured by our current level of approximation. This state is an **EXCITON**.

## Adiabatic LDA (ALDA) approximation
Our first attempt to go beyond RPA is using TDDFT in the Adiabatic LDA approximation. Depending on the exchange-correlation (xc) functional used in the ground state calculation yambo will produce a corresponding xc-kernel. In this case, we are using a Perdew Wang parametrization.
Create a `04_alda_g_space` file
```bash
optics                       # [R OPT] Optics
chi                          # [R LR] Linear Response.
alda_fxc                     # [R TDDFT] The ALDA TDDFT kernel
% QpntsRXd
 1 | 1 |                     # [Xd] Transferred momenta
%
% BndsRnXd
  1 | 10 |                   # [Xd] Polarization function bands
%
NGsBlkXd=  51            RL  # [Xd] Response block size
% EnRngeXd
  7.50000 | 25.00000 |   eV  # [Xd] Energy range
%
% DmRngeXd
 0.100000 | 0.300000 |   eV  # [Xd] Damping range
%
ETStpsXd= 300                # [Xd] Total Energy steps
% LongDrXd
 1.000000 | 0.000000 | 0.000000 |      # [Xd] [cc] Electric Field
%
FxcGRLc= 51              RL  # [TDDFT] XC-kernel RL size
```
and run `yambo` again
```
$ yambo -F 04_alda_g_space -J 04_alda_g_space
```
If we plot again, the result does not improve considerably.

![absorption_tut_v2](https://hackmd.io/_uploads/S1ZlGYWgA.png)

## The Statically screened Electron-electron interaction
As simple LDA and ALDA approximations have not described correctly the experimental spectrum we have tom ove towads more elaborate techniques like the Bethe-Salpeter equation (BSE).
A key ingredient in the BSE kernel is the *electron-electron interaction* commonly evaluated in the static approximation. The input file `05_W` describes how to calculate it
```
em1s                         # [R Xs] Static Inverse Dielectric Matrix
% QpntsRXs
  1 | 19 |                   # [Xs] Transferred momenta
%
% BndsRnXs
  1 | 50 |                   # [Xs] Polarization function bands
%
NGsBlkXs=51              RL  # [Xs] Response block size
% LongDrXs
 1.000000 | 0.000000 | 0.000000 |      # [Xs] [cc] Electric Field
%
```
The variables used in htis input file have the same physical meaning of those used in the optical absorption calculation. The only difference is that, in general, the response function dimension obtained in the examples (03-04), gives an upper bound to the number of RL vectors needed here. This is because the size of response matrix in an RPA calculation defines also the size of the Hartree potential, whose short-range components are not screened. In the present case, instead, the electron-electron interaction is screened and, for this reason, the RL vectors are considerably smaller than in the RPA case.
Run this calculation via
```
$ yambo -F 05_W
```
to create the `ndb.em1s` database which will be read in the BSE calculation.
## The BSE equation, Excitons
In this subsection, we calculate the excitonic absorption spectrum, solving the BSE equation.
1. Create an input file `06_BSE`
```
optics                       # [R OPT] Optics
bse                          # [R BSK] Bethe Salpeter Equation.
bss                          # [R BSS] Bethe Salpeter Equation solver
% KfnQP_E
 5.80000 | 1.000000 | 1.000000 |      # [EXTQP BSK BSS] E parameters (c/v), scissor
%
BSKmod= "SEX"              # [BSK] Resonant Kernel mode. (`x`;`c`;`d`)
% BSEBands
  2 |  7 |                   # [BSK] Bands range
%
BSENGBlk=  51            RL  # [BSK] Screened interaction block size
BSENGexx=  773           RL  # [BSK] Exchange components
BSSmod= "d"                  # [BSS] Solvers `h/d/i/t`
% BEnRange
  7.50000 | 25.00000 |   eV  # [BSS] Energy range
%
% BDmRange
  0.15000 |  0.30000 |   eV  # [BSS] Damping range
%
BEnSteps= 300                # [BSS] Energy steps
% BLongDir
 1.000000 | 0.000000 | 0.000000 |      # [BSS] [cc] Electric Field
%
% BSEQptR
 1 | 19 |                             # [BSK] Transferred momenta range
%
```
Note that we are applying a 5.8 eV scissor and including both exchange and correlations terms in the kernel via the screened exchange (SEX) approximation for the kernel (`BSKmod="SEX"`)
Remember that the BSE kernel is written in Bloch space and its size is given by
```
BSE kernel size = Valence Bands x Conduction bands x K-points in the whole BZ
```

2. Run yambo:
```
$ yambo -F 06_BSE -J 06_BSE
```
3. Plot the output files and compare the results.
```python
...
data_06_BSE = np.loadtxt(f'{YAMBO_TUT_PATH}/shifted-grid/o-06_BSE.eps_q1_diago_bse', usecols=(0,1))
...
ax.plot(data_06_BSE[:,0], data_06_BSE[:,1], label='06_BSE')
```

![absorption_tut_v3](https://hackmd.io/_uploads/r1f3rFbgC.png)

We see that we have much better agreement with experiment and the BSE equation is able to describe the bound electron-hole state responsible for the peak observed experimentally below the QP gap.

## shifted vs unshifted grid
We can now repeat the same steps starting from an unshifted grid (`4 4 4 0 0 0`).
Note that in the unshifted case you have 64 k-points in the BZ, while in the shifted case you have 4 shifts for a total of 64*4=256 points in the BZ. 
![absorptionshitvsunshift](https://hackmd.io/_uploads/BydV9YWeR.png)

The two results differ only slightly.

# Exciton band structure
In this section, we will try to reproduce the exciton band structure from *Neaton et al.* [^3]


<figure>
    <img src="https://hackmd.io/_uploads/r1zOnKWxA.png" alt="Alt text for the image" style="width:60%">
    <figcaption>Figure 1: Caption for the image.</figcaption>
</figure>


# Bibliography
[^1]: Kohn, W. (2019). Density functional theory. Introductory quantum mechanics with MATLAB: for atoms, molecules, clusters, and nanocrystals.
[^2]: Marzari, N., Mostofi, A. A., Yates, J. R., Souza, I., & Vanderbilt, D. (2012). Maximally localized Wannier functions: Theory and applications. Reviews of Modern Physics, 84(4), 1419.
[^3]: Haber, J. B., Qiu, D. Y., da Jornada, F. H., & Neaton, J. B. (2023). Maximally localized exciton Wannier functions for solids. Physical Review B, 108(12), 125118
[^4]: Sangalli, D., Ferretti, A., Miranda, H., Attaccalite, C., Marri, I., Cannuccia, E., ... & Marini, A. (2019). Many-body perturbation theory calculations using the yambo code. Journal of Physics: Condensed Matter, 31(32), 325902.
[^5]: Marini, A., Hogan, C., Grüning, M., & Varsano, D. (2009). Yambo: an ab initio tool for excited state calculations. Computer Physics Communications, 180(8), 1392-1403.
[^6]: Giannozzi, P., Baseggio, O., Bonfà, P., Brunato, D., Car, R., Carnimeo, I., ... & Baroni, S. (2020). Quantum ESPRESSO toward the exascale. The Journal of chemical physics, 152(15).