#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;


void dgemm_sequential(int M, int N, int K, double *A, double *B, double *C) {
  for (int i = 0; i < M; i++) {
    for (int p = 0; p < N; p++) {
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + p];
      }
      C[i * N + p] = sum;
    }
  }
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
    int M, K;
    int N_values[3] = {500, 1000, 1500};
    int P_values[5] = {1, 2, 4, 8, 16};

    // Prompt the user to input the values of M, N, K, and P
    // cout << "Enter the number of rows (M): ";
    // cin >> M;
    M = 100;

    // cout << "Enter the number of columns of A/rows of B (K): ";
    // cin >> K;
    K = 200;

    double time_parallel_stats[3][5];
    double time_sequential_stats[3][5];
    double gflops_parallel_stats[3][5];
    double gflops_sequential_stats[3][5];

    //  Print output table header
    cout << setw(20) << "N" << setw(20) << "P" << setw(20) << "Time (parallel)" << setw(20) << "Time (sequential)" << setw(20) << "GFlops (parallel)" << setw(20) << "GFlops (sequential)" << endl;
    for (int l = 0; l < 6; l++) {
        cout << setw(20) << "-------------------";
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

            // // Print the results
            // cout << "Time (parallel): " << time_parallel << " seconds" << endl;
            // cout << "Time (sequential): " << time_sequential << " seconds" << endl;
            // cout << "GFlops (parallel): " << gflops_parallel << endl;
            // cout << "GFlops (sequential): " << gflops_sequential << endl;

            // Print table row
            cout << setw(20) << N << setw(20) << P << setw(20) << time_parallel << setw(20) << time_sequential << setw(20) << gflops_parallel << setw(20) << gflops_sequential << endl;

        }
        // Free up memory
        free(A);
        free(B);
        free(C_parallel);
        free(C_sequential);
    }

    return 0;
}