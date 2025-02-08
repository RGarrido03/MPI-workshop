#!/bin/bash

#SBATCH -p fct
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4
#SBATCH --qos=uvlabuaveiro

module load python/3.10
module load gcc13/openmpi/4.1.6

source ../venv/bin/activate
mpiexec -np "$SLURM_NTASKS" python array.py
