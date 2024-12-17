#include <iostream>
#include <cstdlib>
// #include <cuda_runtime.h>
// #include "matrix_multiply.cuh"
#include <omp.h>

#define SEED 42 
#define N 1024 // Размер матриц NxN

// Логика умножения матриц через CUDA kernell (matrix_multiply.cu)
extern "C" void multiplyMatrices(double *h_A, double *h_B, double *h_C, int n);

// Логика умножения матриц через CUDA kernell с помощью Pinned памяти (matrix_multiply.cu)
extern "C" void pinnedMultiplyMatrices(double *A, double *B, double *C, int n);

// Type definition for the matrix generation function pointer
typedef void (*MatrixGenFunc)(double*, int);

// Логика умножения матриц через CUDA kernell с помощью Unified памяти
extern "C" void unifiedMultiplyMatrices(MatrixGenFunc matrixGenFunc, int n);

// Логика умножения матриц через CUDA kernell с помощью CUDA-streams
extern "C" void streamingMultiplyMatrices(double *A, double *B, double *C, int n);

// Функция для генерации случайной матрицы
void generateRandomMatrix(double* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// Функция для умножения матриц на CPU
void SequentialSquareMatrixMul(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Функция для умножения матриц на CPU с использованием OpenMP
void ParallelSquareMatrixMul(double* A, double* B, double* C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+: sum)
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
                
            }
            // #pragma omp critical
            C[i * n + j] = sum;
        }   
    }
}

void baseOpenMpPerformance(){
    double *A, *B, *C;

    // Allocate memory for matrices
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));

    // Генерация случайных матриц
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Start timing
    double start_time = omp_get_wtime();

    // Perform matrix multiplication
    ParallelSquareMatrixMul(A, B, C, N);

    // End timing
    double end_time = omp_get_wtime();

    // Calculate and print the elapsed time
    std::cout << "OpenMP Matrix multiplication time: " << (end_time - start_time) * 1000 << " ms" << std::endl;

    // Clean up
    free(A);
    free(B);
    free(C);
}

void task_1() {
    std::cout << "Task #1: Перемножение матриц с использованием глобальной памяти на CUDA" << "\n";

    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;

    // Выделение памяти на CPU
    h_A = (double*)malloc(N * N * sizeof(double));
    h_B = (double*)malloc(N * N * sizeof(double));
    h_C = (double*)malloc(N * N * sizeof(double));

    // Генерация случайных матриц
    generateRandomMatrix(h_A, N);
    generateRandomMatrix(h_B, N);

    // Запускаем умножение на GPU
    multiplyMatrices(h_A, h_B, h_C, N);

    // Освобождение памяти на CPU
    free(h_A);
    free(h_B);
    free(h_C);
} 

void task_2_a() {
    std::cout << "Task #2 - a: использование Pinned памяти" << "\n";

    double *A, *B, *C;

    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));

    // Генерация случайных матриц
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Запускаем умножение на GPU
    pinnedMultiplyMatrices(A, B, C, N);
    
    free(A);
    free(B);
    free(C);
}

void task_2_b(){
    std::cout << "Task #2 - b: работа с памятью передана Unified Memory" << "\n";
    
    // Запускаем умножение на GPU
    unifiedMultiplyMatrices(generateRandomMatrix, N);
}

void task_2_c(){
    std::cout << "Task #2 - c: работа с памятью передана CUDA-streams" << "\n";
    
    double *A, *B, *C;
    
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));

    // Генерация случайных матриц
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Запускаем умножение на GPU
    streamingMultiplyMatrices(A, B, C, N);

    free(A);
    free(B);
    free(C);
}

void task_2() {
    std::cout << "Task #2:  Оптимизировать работу с памятью" << "\n";
    task_2_a();
    task_2_b();
    task_2_c();
}



int main() {
    // Воспроизводимость случайной матрицы
    srand(SEED);

    std::cout << "Using square matrices of size " << N << "x" << N << "\n";
    baseOpenMpPerformance();
    task_1();
    task_2();
    return 0;
}