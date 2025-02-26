#!/bin/bash

#SBATCH -p hpc
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=normal

module load python/3.10
module load udocker/1.3.17

udocker load -i ~/MPI-workshop/ai/mpi-ruben.tar mpi-ruben
udocker create --name=mpi-workshop mpi-ruben
