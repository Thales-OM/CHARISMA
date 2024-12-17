#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;


// Simple sequential imlementation
void dgemm_sequential(int M, int N, int K, double *A, double *B, double *C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

//My parallel dgemm implementation
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

namespace openblas {
    #include <cblas.h>

    // Function to perform matrix multiplication using OpenBLAS
    void openblas_dgemm(int M, int N, int K, double* A, double* B, double* C) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    }
}

namespace mkl {
    #include <mkl.h>

    // Function to perform matrix multiplication using MKL
    void mkl_dgemm(int M, int N, int K, double* A, double* B, double* C) {
        double alpha = 1.0;
        double beta = 0.0;
        char transa = 'N';
        char transb = 'N';

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
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

void print_matrix(double* A, int M, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", A[i * K + j]);
        }
        printf("\n");
    }
}



void chech_that_dgemm_works(int M, int N, int K) {
    cout << endl << "Task #1: Chech correctness of my dgemm parallel implementation" << endl;
    cout << "Using: M = " <<  M << ", N = " << N << ", K = " << K << endl;

    // Allocate memory for the matrices
    double *A = (double *)malloc(M * K * sizeof(double));
    double *B = (double *)malloc(K * N * sizeof(double));
    double *C_parallel = (double *)malloc(M * N * sizeof(double));
    double *C_sequential = (double *)malloc(M * N * sizeof(double));

    // Initialize the matrices with some values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = i + j;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = i - j;
        }
    }

    // Print the input matrices
    cout << "Input matrix A:\n";
    print_matrix(A, M, K);
    cout << endl;
    cout << "Input matrix B:\n";
    print_matrix(B, K, N);
    cout << endl;

    // Call parallel implementation
    blas_dgemm(M, N, K, A, B, C_parallel);
    // Call sequential implementation
    dgemm_sequential(M, N, K, A, B, C_sequential);


    // Print the result matrix (paralellel)
    printf("Result matrix C (parallel implementation):\n");
    print_matrix(C_parallel, M, N);

    // Print the result matrix
    printf("Result matrix C (sequential implementation):\n");
    print_matrix(C_sequential, M, N);

    if (!compare_same_size_matrices(C_parallel, C_sequential, M, N)) {
        throw runtime_error("Sequential an parallel implementation returned different matrices!");
    } else {
        cout << "Sequential and parallel implementation returned the same matrix!" << endl;
    }

    // Free up memory
    free(A);
    free(B);
    free(C_parallel);
    free(C_sequential);
}

void compare_parallel_to_sequential(int M, int K) {
    cout << endl << "Task #2: Analyze scalability of parallel implementation using a sequential one" << endl;
    cout << "Using: M = " <<  M << ", K = " << K << endl;
    
    int N_values[3] = {500, 1000, 1500};
    int P_values[5] = {1, 2, 4, 8, 16};

    double time_parallel_stats[3][5];
    double time_sequential_stats[3][5];
    double gflops_parallel_stats[3][5];
    double gflops_sequential_stats[3][5];

    //  Print output table header
    cout << setw(25) << "N" << setw(25) << "P" << setw(25) << "Time (parallel), sec" << setw(25) << "Time (sequential), sec" << setw(25) << "GFlops (parallel)" << setw(25) << "GFlops (sequential)" << endl;
    for (int l = 0; l < 6; l++) {
        cout << setw(25) << "-----------------------";
    }
    cout << endl;

    for (int i = 0; i < 3; i++) {
        int N = N_values[i];
            
        // Allocate memory for the matrices
        double *A = (double *)malloc(M * K * sizeof(double));
        double *B = (double *)malloc(K * N * sizeof(double));
        double *C_parallel = (double *)malloc(M * N * sizeof(double));
        double *C_sequential = (double *)malloc(M * N * sizeof(double));

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

        for (int j = 0; j < 5; j++) {
            int P = P_values[j];
            
            // Set the number of threads
            omp_set_num_threads(P);

            // Measure the time for the parallel implementation
            auto start_time_parallel = chrono::high_resolution_clock::now();
            blas_dgemm(M, N, K, A, B, C_parallel);
            auto end_time_parallel = chrono::high_resolution_clock::now();
            double time_parallel = chrono::duration_cast<chrono::microseconds>(end_time_parallel - start_time_parallel).count() / 1e6;
            time_parallel_stats[i][j] = time_parallel;

            // Measure the time for the sequential implementation
            auto start_time_sequential = chrono::high_resolution_clock::now();
            dgemm_sequential(M, N, K, A, B, C_sequential);
            auto end_time_sequential = chrono::high_resolution_clock::now();
            double time_sequential = chrono::duration_cast<chrono::microseconds>(end_time_sequential - start_time_sequential).count() / 1e6;
            time_sequential_stats[i][j] = time_sequential;

            // // Print the result matrix (paralellel)
            // printf("Result matrix C (parallel implementation):\n");
            // for (int i = 0; i < M; i++) {
            //     for (int j = 0; j < N; j++) {
            //         printf("%f ", C_parallel[i * N + j]);
            //     }
            //     printf("\n");
            // }

            // // Print the result matrix
            // printf("Result matrix C (sequential implementation):\n");
            // for (int i = 0; i < M; i++) {
            //     for (int j = 0; j < N; j++) {
            //         printf("%f ", C_sequential[i * N + j]);
            //     }
            //     printf("\n");
            // }

            if (!compare_same_size_matrices(C_parallel, C_sequential, M, N)) {
                throw runtime_error("Sequential an parallel implementation returned different matrices!");
            }

            // Calculate the GFlops for both implementations
            double gflops_parallel = (2.0 * M * N * K) / (time_parallel * 1e9);
            gflops_parallel_stats[i][j] = gflops_parallel;
            double gflops_sequential = (2.0 * M * N * K) / (time_sequential * 1e9);
            gflops_sequential_stats[i][j] = gflops_sequential;

            // Print table row
            cout << setw(20) << N << setw(20) << P << setw(20) << time_parallel << setw(20) << time_sequential << setw(20) << gflops_parallel << setw(20) << gflops_sequential << endl;

        }
        // Free up memory
        free(A);
        free(B);
        free(C_parallel);
        free(C_sequential);
    }
}

void compare_mine_to_openblas(int M, int N, int K) {
    cout << endl << "Task #3: Compare my dgemm implementation to OpenBlas, MKL" << endl;
    cout << "Using: M = " <<  M << ", N = " << N << ", K = " << K << endl;
    
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
    blas_dgemm(M, N, K, A, B, C);
    auto end_time_parallel = chrono::high_resolution_clock::now();
    double time_mine = chrono::duration_cast<chrono::microseconds>(end_time_parallel - start_time_parallel).count() / 1e6;

    // Measure the execution time of OpenBLAS
    auto start_time_openblas = std::chrono::high_resolution_clock::now();
    openblas::openblas_dgemm(M, N, K, A, B, C_openblas);
    auto end_time_openblas = std::chrono::high_resolution_clock::now();
    double time_openblas = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_openblas - start_time_openblas).count() / 1e6;

    // Measure the execution time of MKL
    auto start_time_mkl = std::chrono::high_resolution_clock::now();
    mkl::mkl_dgemm(M, N, K, A, B, C_mkl);
    auto end_time_mkl = std::chrono::high_resolution_clock::now();
    double time_mkl = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_mkl - start_time_mkl).count() / 1e6;

    // Calculate the GFlops for all implementations
    double gflops_mine = (2.0 * M * N * K) / (time_mine * 1e9);
    double gflops_openblas = (2 * M * N * K) / (time_openblas * 1e9);
    double gflops_mkl = (2 * M * N * K) / (time_mkl * 1e9);

    // // Print the input matrices
    // cout << "Input matrix A:\n";
    // print_matrix(A, M, K);
    // cout << endl;
    // cout << "Input matrix B:\n";
    // print_matrix(B, K, N);
    // cout << endl;
    // cout << "My matrix C:\n";
    // print_matrix(C, M, N);
    // cout << endl;
    // cout << "OpenBlas matrix C:\n";
    // print_matrix(C_openblas, M, N);
    // cout << endl;
    // cout << "MKL matrix C:\n";
    // print_matrix(C_mkl, M, N);
    // cout << endl;

    if (!compare_same_size_matrices(C, C_openblas, M, N)) {
        throw runtime_error("Mine and OpenBLAS implementation returned different matrices!");
    }

    if (!compare_same_size_matrices(C, C_mkl, M, N)) {
        throw runtime_error("Mine and MKL implementation returned different matrices!");
    }

    //  Print output table header
    cout << setw(20) << "Version" << setw(20) << "Time, sec" << setw(20) << "GFlops" << endl;
    for (int l = 0; l < 3; l++) {
        cout << setw(20) << "-------------------";
    }
    cout << endl;
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
}

int main() {
    chech_that_dgemm_works(3,5,4);
    compare_parallel_to_sequential(1000,1000);
    compare_mine_to_openblas(1000, 2000, 1500);

    return 0;
}