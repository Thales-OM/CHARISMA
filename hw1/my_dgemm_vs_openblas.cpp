#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cblas.h>
#include <mkl.h>

using namespace std;


// Function to perform matrix multiplication using OpenBLAS
void openblas_dgemm(double* A, double* B, double* C, int M, int N, int K) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
}

// Function to perform matrix multiplication using MKL
void mkl_dgemm(double* A, double* B, double* C, int M, int N, int K) {
    mkl_dgemm("N", "N", &M, &N, &K, &1.0, A, &K, B, &N, &0.0, C, &N);
}

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C) {
//   double dot_product[M][N];
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int p = 0; p < N; p++) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+: sum)
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + p];
        }
        // #pragma omp critical
        C[i * N + p] = sum;
    }
  }
}

bool compare_same_size_matrices(double* A, double* B, int M, int N) {
    // Compare contents
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            if (A[index] != B[index]) {
                return false; // matrices have different contents
            }
        }
    }
    return true; // matrices are equal
}

int main() {
    int M, K, N;

    // Prompt the user to input the values of M, N, K, and P
    // cout << "Enter the number of rows (M): ";
    // cin >> M;
    M = 100;

    // cout << "Enter the number of columns of A/rows of B (K): ";
    // cin >> K;
    K = 200;

    // cout << "Enter the number of columns (N): ";
    // cin >> N;
    N = 300;

    //  Print output table header
    cout << setw(20) << 'Version' << setw(20) << "Time, sec" << setw(20) << "GFlops" << endl;
    for (int l = 0; l < 3; l++) {
        cout << setw(20) << "-------------------";
    }
    cout << endl;
            
    // Allocate memory for the matrices
    double *A = (double *)malloc(M * K * sizeof(double));
    double *B = (double *)malloc(K * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));
    double* C_openblas = (double *)malloc(M * N * sizeof(double));
    double* C_mkl = (double *)malloc(M * N * sizeof(double));

    // Initialize the matrices with some values
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
        A[m * K + k] = m + k;
        }
    }

    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
        B[k * N + n] = k - n;
        }
    }

    // Measure the time of my implementation
    auto start_time_parallel = chrono::high_resolution_clock::now();
    blas_dgemm(M, N, K, A, B, C_parallel);
    auto end_time_parallel = chrono::high_resolution_clock::now();
    double time_mine = chrono::duration_cast<chrono::microseconds>(end_time_parallel - start_time_parallel).count() / 1e6;

    // Measure the execution time of OpenBLAS
    auto start_time_openblas = std::chrono::high_resolution_clock::now();
    openblas_dgemm(A, B, C_openblas, M, N, K);
    auto end_time_openblas = std::chrono::high_resolution_clock::now();
    double time_openblas = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_openblas - start_time_openblas).count() / 1e6;

    // Measure the execution time of MKL
    auto start_time_mkl = std::chrono::high_resolution_clock::now();
    mkl_dgemm(A, B, C_mkl, M, N, K);
    auto end_time_mkl = std::chrono::high_resolution_clock::now();
    double time_mkl = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_mkl - start_time_mkl).count() / 1e6;

    // Calculate the GFlops for all implementations
    double gflops_mine = (2.0 * M * N * K) / (time_mine * 1e9);
    double gflops_openblas = (2 * M * N * K) / (time_openblas * 1e9);
    double gflops_mkl = (2 * M * N * K) / (time_mkl * 1e9);


    if (!compare_same_size_matrices(C, C_openblas, M, N)) {
        throw runtime_error("Mine and OpenBLAS implementation returned different matrices!");
    }

    if (!compare_same_size_matrices(C, C_mkl, M, N)) {
        throw runtime_error("Mine and MKL implementation returned different matrices!");
    }

    // Print table row for each implementation
    cout << setw(20) << "Mine" << setw(20) << time_mine << setw(20) << gflops_mine << endl;
    cout << setw(20) << "OpenBLAS" << setw(20) << time_openblas << setw(20) << gflops_openblas << endl;
    cout << setw(20) << "MKL" << setw(20) << time_mkl << setw(20) << gflops_mkl << endl;
            
    // Free up memory
    free(A);
    free(B);
    free(C);
    free(C_openblas);
    free(C_mkl);

    return 0;
}