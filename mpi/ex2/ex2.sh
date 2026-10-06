#!/bin/bash

#SBATCH -p dev-x86
#SBATCH -A F202316480ICDTF2X
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=512M

module load Python/3.14.2-GCCcore-15.2.0
module load OpenMPI/5.0.10-GCC-15.2.0

source ../.venv/bin/activate
mpiexec -np "$SLURM_NTASKS" python ex2.py
