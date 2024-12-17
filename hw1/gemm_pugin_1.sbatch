#!/bin/bash

#SBATCH --job-name=gemm_pugin_1
#SBATCH --output=gemm_pugin_1%j.out
#SBATCH --error=gemm_pugin_1%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH --constraint="type_d"

module load gcc/9.3.0
module load openmpi/4.1.2
module load INTEL/oneAPI_2021_u2
module load mkl/latest
module load OpenBlas/v0.3.18

gcc -lstdc++ -fopenmp -lopenblas -o gemm_pugin_1 gemm_pugin_1.cpp
srun ./gemm_pugin_1