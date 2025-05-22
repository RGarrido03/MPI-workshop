#!/bin/bash

#SBATCH -p tuthpc
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

export MODULEPATH="/etc/scl/modulefiles:/cvmfs/sw.el8/modules/hpc/main:/cvmfs/sw.el8/modules/hpc/aoc40:/cvmfs/sw.el8/modules/hpc/gcc85:/cvmfs/sw.el8/modules/hpc/gcc11:/cvmfs/sw.el8/modules/hpc/gcc13:/cvmfs/sw.el8/modules/hpc/intel:/cvmfs/sw.el8/modules/gpu:/cvmfs/sw.el8/modules/ml:/cvmfs/sw.el8/modules/bio"

module load python/3.10
module load gcc13/openmpi/4.1.6

source ../venv/bin/activate
mpiexec -np "$SLURM_NTASKS" python ex2.py
