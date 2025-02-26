#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --qos=normal

module load python/3.10
module load udocker/1.3.17
module load cuda/12.6

udocker setup --nvidia mpi-workshop
udocker run -v ~/MPI-workshop/ai:/app mpi-workshop
