#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>

// #include "matrix_multiply.cuh"


#define TILE_SIZE 16 // Размер блока (TILE_SIZE x TILE_SIZE)

// CUDA Kernel для умножения матриц
__global__ void squareMatrixMultiply(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Индекс строки
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Индекс столбца

    // Проверка, что не вылетели за границы итоговой матрицы
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[k * n + col] * B[row * n + k]; // Умножение с учетом column-major порядка
        }
        C[row * n + col] = sum;
    }
}

// Логика умножения матриц через CUDA kernell
extern "C" void multiplyMatrices(double *h_A, double *h_B, double *h_C, int n) {
    double *d_A, *d_B, *d_C;

    // Выделение памяти на GPU
    cudaMalloc((void **)&d_A, n * n * sizeof(double));
    cudaMalloc((void **)&d_B, n * n * sizeof(double));
    cudaMalloc((void **)&d_C, n * n * sizeof(double));

    // Создание событий для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запуск события перед выполнением ядра
    cudaEventRecord(start);

    // Копирование матриц A и B на GPU
    cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Вызов ядра
    squareMatrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Копирование результата обратно на CPU
    cudaMemcpy(h_C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Запуск события после выполнения ядра
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Измерение времени выполнения
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken for matrix multiplication (CUDA): %.2f ms\n", elapsedTime);

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Логика умножения матриц через CUDA kernell с помощью Pinned памяти
extern "C" void pinnedMultiplyMatrices(double *A, double *B, double *C, int n) {
    cudaEvent_t start, stop;
    float elapsedTime;
    double *d_A, *d_B, *d_C;

    cudaMallocHost(&A, n * n * sizeof(double));
    cudaMallocHost(&B, n * n * sizeof(double));
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_B, n * n * sizeof(double));
    cudaMalloc(&d_C, n * n * sizeof(double));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyAsync(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = TILE_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    squareMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    // cudaDeviceSynchronize();
    
    cudaMemcpyAsync(C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time taken for matrix multiplication (CUDA): %.2f ms\n", elapsedTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Type definition for the matrix generation function pointer
typedef void (*MatrixGenFunc)(double*, int);

// Логика умножения матриц через CUDA kernell с помощью Unified памяти
extern "C" void unifiedMultiplyMatrices(MatrixGenFunc matrixGenFunc, int n) {
    double *A, *B, *C;
    cudaMallocManaged(&A, n * n * sizeof(double));
    cudaMallocManaged(&B, n * n * sizeof(double));
    cudaMallocManaged(&C, n * n * sizeof(double));

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Генерация случайных матриц
    matrixGenFunc(A, n);
    matrixGenFunc(B, n);

    cudaEventRecord(start);

    int threadsPerBlock = TILE_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    squareMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);
    // cudaDeviceSynchronize();


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time taken for matrix multiplication (CUDA): %.2f ms\n", elapsedTime);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

// Логика умножения матриц через CUDA kernell с помощью CUDA-streams
extern "C" void streamingMultiplyMatrices(double *A, double *B, double *C, int n) {
    double *d_A, *d_B, *d_C;
    cudaEvent_t start, stop;
    cudaStream_t stream1, stream2, stream3, stream4;

    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_B, n * n * sizeof(double));
    cudaMalloc(&d_C, n * n * sizeof(double));
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyAsync(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice, stream2);

    int threadsPerBlock = TILE_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    squareMatrixMultiply<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(d_A, d_B, d_C, n);

    cudaMemcpyAsync(C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost, stream4);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Wait for the stream to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time taken for matrix multiplication (CUDA): %.2f ms\n", elapsedTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
}