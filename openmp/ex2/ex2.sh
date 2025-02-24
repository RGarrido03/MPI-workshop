#!/bin/bash

#SBATCH -p fct
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --qos=uvlabuaveiro

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

module load gcc-14.1
./ex2
