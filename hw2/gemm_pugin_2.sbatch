#!/bin/bash

#SBATCH --job-name=gemm_pugin_2
#SBATCH --output=gemm_pugin_2%j.out
#SBATCH --error=gemm_pugin_2%j.err

module load openmpi/4.1.2
module load nvidia_sdk/nvhpc/23.5 
module load CUDA/12.2

nvcc -c matrix_multiply.cu -o matrix_multiply.o
g++ -fopenmp -lcudart gemm_pugin_2.cpp matrix_multiply.o -o gemm_pugin_2
salloc -n 1 --gpus=1 -Aproj_1593
srun ./gemm_pugin_2