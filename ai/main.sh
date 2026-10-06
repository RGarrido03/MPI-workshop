#!/bin/bash

#SBATCH -p dev-a100-40
#SBATCH -A F202316480ICDTF2G
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

mpirun -np 4 singularity exec container.sif python3 main.py
