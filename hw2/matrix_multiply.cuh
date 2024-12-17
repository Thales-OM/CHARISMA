#ifndef MATRIX_MULTIPLY_CUH
#define MATRIX_MULTIPLY_CUH

#include <cuda_runtime.h>

__global__ void squareMatrixMultiply(double *A, double *B, double *C, int n);

#endif // MATRIX_MULTIPLY_CUH