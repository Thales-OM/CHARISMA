#!/bin/bash

#SBATCH --job-name=pugin_3
#SBATCH --output=pugin_3%j.out
#SBATCH --error=pugin_3%j.err
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --constraint="type_d"

module load openmpi/4.1.5_cuda12
module load nvidia_sdk/nvhpc/23.5 
module load CUDA/12.2
module load hpcx/hpcx-cuda-ompi

mpicxx pugin_3_v1.cpp -o pugin_3_v1.o
srun --ntasks=6 --cpus-per-task=1 --nodes=2 --constraint="type_d" --time=00:10:00 ./pugin_3_v1.o