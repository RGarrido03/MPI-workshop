#!/bin/bash

#SBATCH -p hpc
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --qos=normal

#
# Build using:
# module load gcc-14.1
# cmake --build .
#
# or:
# module load gcc-14.1
# cmake .
# make
#

export MODULEPATH="/etc/scl/modulefiles:/cvmfs/sw.el8/modules/hpc/main:/cvmfs/sw.el8/modules/hpc/aoc40:/cvmfs/sw.el8/modules/hpc/gcc85:/cvmfs/sw.el8/modules/hpc/gcc11:/cvmfs/sw.el8/modules/hpc/gcc13:/cvmfs/sw.el8/modules/hpc/intel:/cvmfs/sw.el8/modules/gpu:/cvmfs/sw.el8/modules/ml:/cvmfs/sw.el8/modules/bio"

module load gcc-14.1
./ex1
