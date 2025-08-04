# Quantum ESPRESSO - Compilation Guide
Quantum ESPRESSO (QE) is generally straightforward to compile, especially on modern HPC systems where dependencies are available via modules. In most cases, you may not even need to compile it yourself, as precompiled versions are often provided.

For example, on **Snellius** (the Dutch National Supercomputer), you can simply load a precompiled module:

```
module load QuantumESPRESSO/7.1-foss-2022a
```
If you need to compile QE yourself (e.g., to enable specific options like EPW or OpenMP), follow the instructions below.

0. Clone the QE source code from the official [Quantum ESPRESSO Foundation webpage](https://gitlab.com/QEF/q-e).

```
git clone https://gitlab.com/QEF/q-e.git
cd q-e
```
It is strongly recommended to check out a stable release, which you can find in the repository's Releases section. For example, `git checkout qe-7.4`
2. Create a job script (`compile_qe.sh`) to compile QE. Below is an example SLURM script used on Snellius:

```
#!/bin/bash -l
#SBATCH -J "qe_ph"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --hint=multithread
#SBATCH -p  genoa


module load 2022 \
foss/2022a \

./configure --enable-openmp

make -j2 all
make -j2 epw
```
3. Submit the script using:
```
sbatch compile_qe.slurm
```

# Notes
- Adjust --cpus-per-task and make -j according to your node's hardware.

- If you plan to use specific libraries (e.g., MKL, MPI wrappers, HDF5, etc.), you may need to configure QE accordingly using flags like --with-libxc or environment variables such as F90, CC, etc.

- For additional components like XSpectra, GWL, or EPW, remember to compile them explicitly if not included in the default make all.