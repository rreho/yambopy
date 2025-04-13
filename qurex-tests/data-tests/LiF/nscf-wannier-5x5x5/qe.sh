#!/bin/bash
##SBATCH --nodes=1              # Number of Nodesi, 128 cpu per node
#SBATCH --ntasks=12  # Number of MPI tasks per node
#SBATCH --cpus-per-task=1       # Number of OpenMP threads
##SBATCH --hint=nomultithread    # Disable hyperthreading
#SBATCH --exclusive
#SBATCH --time=00:10:00         #Expected runtime HH:MM:SS (max 100h)
#SBATCH --mail-type=ALL
##SBATCH --mail-user=r.reho@uu.nl
##
#SBATCH --exclusive
# Print environment
export OMP_PROC_BIND='false'
# Manage modules
source ~/modules_qe.load
# Execute commands
srun /home/rireho/codes/q-e/branches/qe-7.1/bin/pw.x < nscf.in > log_nscf.out
