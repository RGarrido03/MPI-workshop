#!/bin/bash

#SBATCH -p hpc
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --qos=normal

module load python/3.10
module load gcc13/openmpi/4.1.6

source venv/bin/activate
python main.py
