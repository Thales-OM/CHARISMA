#!/bin/bash

#SBATCH --n-tasks=6
#SBATCH --job-name=pugin_3
#SBATCH --output=pugin_3%j.out
#SBATCH --error=pugin_3%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --constraint="type_d"

module load gnu8 
module load openmpi3

mpicxx pugin_3_v1.cpp -o pugin_3_v1.o
srun --ntasks=6 --cpus-per-task=1 --nodes=2 --constraint="type_d" --time=00:10:00 ./pugin_3_v1.o