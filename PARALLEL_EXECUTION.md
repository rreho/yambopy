# Instructions for parallel executions

Yambopy supports extremely basic parallel execution.

## Threads | openMP | shared memory
Yambopy inherits the multithreading support of numpy. 

If you select
```
export OMP_NUM_THREADS=nthreads
```
then numpy will automatically take advantage of this. 
Notice however that this is not always guaranteed to speed up execution.

## Tasks | openMPI/mpich | distributed memory
Very few functions support MPI parallelization via the package `mpi4py`.
It is useless if you don't plan on running them.

MPI-supporting functions:
1. `exciton_phonon_matelem`

Take care when installing the `mpi4py` package that the correct environment is used.
For example, for mpirun/mpiexec (i.e., openMPI), you can install it as
```
conda install -c conda-forge mpi4py openmpi
```
Or using `pip` as
```
pip install mpi4py openmpi
```

## Tasks | pipeline jobs with joblib
Very few functions support process-based parallelization via the package `joblib`.

`joblib`-supporting functions:
1. `exc_ph_luminescence`

