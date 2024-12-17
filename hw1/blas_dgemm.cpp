#include <stdio.h>
#include <stdlib.h>
// #include <omp.h>
#include <iostream>

using namespace std;


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

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C) {
    #pragma omp parallel for collapse(2) reduction(+:C[:M*N])
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
    int M, N, K;

    // Prompt the user to input the values of M, N, and K
    // cout << "Enter the number of rows (M): ";
    // cin >> M;
    M = 10;
    // cout << "Enter the number of columns (N): ";
    // cin >> N;
    N = 30;
    // cout << "Enter the number of columns of A/rows of B (K): ";
    // cin >> K;
    K =20;

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
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            cout << A[i * K + j] << " ";
        }
        cout << "\n";
    }

    cout << "Input matrix B:\n";
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            cout << B[i * N + j] << " ";
        }
        cout << "\n";
    }


    // Call parallel implementation
    blas_dgemm(M, N, K, A, B, C_parallel);
    // Call sequential implementation
    dgemm_sequential(M, N, K, A, B, C_sequential);


    // Print the result matrix (paralellel)
    printf("Result matrix C (parallel implementation):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C_parallel[i * N + j]);
        }
        printf("\n");
    }

    // Print the result matrix
    printf("Result matrix C (sequential implementation):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C_sequential[i * N + j]);
        }
        printf("\n");
    }

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

    return 0;
}