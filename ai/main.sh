#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1

export MODULEPATH="/etc/scl/modulefiles:/cvmfs/sw.el8/modules/hpc/main:/cvmfs/sw.el8/modules/hpc/aoc40:/cvmfs/sw.el8/modules/hpc/gcc85:/cvmfs/sw.el8/modules/hpc/gcc11:/cvmfs/sw.el8/modules/hpc/gcc13:/cvmfs/sw.el8/modules/hpc/intel:/cvmfs/sw.el8/modules/gpu:/cvmfs/sw.el8/modules/ml:/cvmfs/sw.el8/modules/bio"

module load python/3.10
module load udocker/1.3.17
module load cuda/12.6

udocker setup --nvidia --force mpi-workshop
udocker run -v ~/MPI-workshop/ai:/app mpi-workshop
