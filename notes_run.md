# 0. Prerequisites

- Run scf, nscf , gw, bse for a given system
- Go here for all binary files /mnt/aiongpfs/projects/tss-physics/software/. In particular for yambo use /mnt/aiongpfs/projects/tss-physics/software/lumen/branches/bug-fixes/bin/- Run dfpt with a grid consistent with nscf and gw bse
- In the same folder where you run the nscf run projwfc.x in a uniform grid 48x48x1
- Run Yambo initialization
- Get this repo: https://github.com/muralidhar-nalabothula/PhdScripts
# 1. GW + BSE

- Run a GW calculation and a BSE calculation of Yambo.
- branch `phys-exc-dispersion` of RR's repo. You can get exciton dispersion, interpolated or not, spin-projected or not, and orbital-projected band structure. To do so, you need to solve the BSE for the full list of Q-points in the IBZ and run 
- For each high-symmetry points report the irrep or irreps of the given exciton state.
`yambopy exc-irrep -J BSE --iqpt 1 --nstates 5 --degen_tol 0.02 --sym_tol 0.1`
This step requires some fine tuning of tolerances for `degen_tol` and `sym_tol`
It might be helpful to visualize the sorted exciton energies.
# 1. LetzElPhC

- Run letzelphc from the variation of the self-consistent potential (dfpt) to get ndb.elph and ndb.Dmats.
How to run is described in doc/main.pdf of LetzElPhC repo. In short,
You have to be in the dfpt folder and create a lelphc.in file.
Run preprocess: `lelphc -pp --code=qe -F PH.X_input_file`
Run lelphc : `mpirun -n 4 lelphc -F lelphc.in`

You get `ndb.elph` and `ndb.Dmats` and you keep them

# 2. Get the exciton-phonon coupling matrix elements

- Check the scripts available here /Users/riccardo.reho/personalcodes/yambopy/tutorial/exciton-phonon and I provide as well other scripts in the root folder if needed.
- Run the `ex_ph_program.py` available in PhDscripts to compute the .npy array for exciton-phonon coupling matrix elements which will be reused by Raman and luminescence. The only difference is that, while Raman, can only be computed with PhDscripts, luminescence has two ways: PhDscripts and yambopy.
- If you want to get `Ex-ph.npy` from yambopy you can run the script `compute_excph.py`. Be careful, different branches might have slightly different arguments and you need to compute it only for Gamma. Hence, no huge arrays.

## 2.1 Raman 

Scripts in PhDscripts.
- Run `raman_driver_ip.py` to compute IP Raman.
- Run `raman_driver.py` to compute exciton-phonon assisted Raman. 

## 2.1 Luminescence (optional)
- Get luminescence. You can get it with `lumin_yambopy_new.py`
