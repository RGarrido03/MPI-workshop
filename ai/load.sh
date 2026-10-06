#!/bin/bash

#SBATCH -p dev-a100-40
#SBATCH -A F202316480ICDTF2G
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

singularity build --fakeroot container.sif container.def
