#!/bin/bash

#SBATCH -p dev-x86
#SBATCH -A F202316480ICDTF2X
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=512M

#
# Build using:
# module load GCC/15.2.0 CMake/4.2.1-GCCcore-15.2.0
# cmake --build .
#
# or:
# module load GCC/15.2.0 CMake/4.2.1-GCCcore-15.2.0
# cmake .
# make
#

srun -n 1 -c 4 ./ex2
