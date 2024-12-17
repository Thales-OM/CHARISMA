#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // char message[20];
    int rank, root = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data;
    if (rank == root) {
        // strcpy(message, "Hello, world!");
        data = 42;
    }

    // int message_length = sizeof(message);

    printf("Process %d before MPI_Bcast\n", rank);
    MPI_Bcast(&data, 1, MPI_INT, root, MPI_COMM_WORLD);
    
    printf("Message from process = %d : %d\n", rank, data);

    MPI_Finalize();
    return 0;
}