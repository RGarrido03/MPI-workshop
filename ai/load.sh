#!/bin/bash

#SBATCH -p hpc
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=normal

export MODULEPATH="/etc/scl/modulefiles:/cvmfs/sw.el8/modules/hpc/main:/cvmfs/sw.el8/modules/hpc/aoc40:/cvmfs/sw.el8/modules/hpc/gcc85:/cvmfs/sw.el8/modules/hpc/gcc11:/cvmfs/sw.el8/modules/hpc/gcc13:/cvmfs/sw.el8/modules/hpc/intel:/cvmfs/sw.el8/modules/gpu:/cvmfs/sw.el8/modules/ml:/cvmfs/sw.el8/modules/bio"

module load python/3.10
module load udocker/1.3.17

udocker load -i ~/MPI-workshop/ai/mpi-ruben.tar mpi-ruben
udocker create --name=mpi-workshop mpi-ruben
