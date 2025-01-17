#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

void tempDistribution(double tau, double t, double h, double* points, int num_points, double* output) {

    int arraySize = static_cast<int>(1 / h) + 1; // Length of the array
    int totalSteps = static_cast<int>(t / tau); // Total number of time steps
    
    MPI_Init(nullptr, nullptr);
    
    int rank, size, root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Starting job" << std::endl;
    }

    // Calculate the chunk size for each process
    int chunkSize = arraySize / size;
    int remainder = arraySize % size;

    // Distribute the remainder to the first few processes
    int startIndex = rank * chunkSize + std::min(rank, remainder);
    int endIndex = startIndex + chunkSize + (rank < remainder ? 1 : 0);

    // Initialize the local array for this process
    std::vector<double> localArray(endIndex - startIndex, 1.0);

    

    // Main time-stepping loop
    for (int step = 0; step < totalSteps; ++step) {
        std::vector<double> nextArray(localArray.size(), 0.0);

        // Prepare to receive neighbor values
        double leftNeighbor = (startIndex > 0) ? localArray[0] : 0.0; // Left neighbor
        double rightNeighbor = (endIndex < arraySize) ? localArray.back() : 0.0; // Right neighbor

        // Send and receive neighbor values
        if (rank > 0) {
            std::cout << "Sending left neighbor by rank " << rank << " to rank " << rank - 1 << std::endl;
            MPI_Send(&localArray[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD); // Send leftmost value to left neighbor
            std::cout << "Finished sending left neighbor by rank " << rank << " to rank " << rank - 1 << std::endl;
        }

        if (rank < size - 1) {
            std::cout << "Recieving right neighbor by rank " << rank << " from rank " << rank + 1 << std::endl;
            MPI_Recv(&rightNeighbor, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive right neighbor
            std::cout << "Finished recieving right neighbor by rank " << rank << " from rank " << rank + 1 << std::endl;
        }

        if (rank < size - 1) {
            std::cout << "Sending right neighbor by rank " << rank << " to rank " << rank + 1 << std::endl;
            MPI_Send(&localArray.back(), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD); // Send rightmost value to right neighbor
            std::cout << "Finished sending right neighbor by rank " << rank << " to rank " << rank + 1 << std::endl;
        }

        if (rank > 0) {
            std::cout << "Recieving left neighbor by rank " << rank << " from rank " << rank - 1 << std::endl;
            MPI_Recv(&leftNeighbor, 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive left neighbor
            std::cout << "Finished recieving left neighbor by rank " << rank << " from rank " << rank - 1 << std::endl;
        }

        // Compute new temperature values
        for (int i = 0; i < localArray.size(); ++i) {
            double left = (i == 0) ? leftNeighbor : localArray[i - 1];
            double right = (i == localArray.size() - 1) ? rightNeighbor : localArray[i + 1];
            nextArray[i] = (left + right) / 2.0;
        }

        localArray = nextArray; // Update local array for the next time step

        std::cout << "Finished step " << step << " on rank " << rank << std::endl;
    }

    // Gather results at the root process
    std::vector<double> finalArray;
    if (rank == root) {
        finalArray.resize(arraySize);
    }

    // Gather all local arrays into the final array
    MPI_Gather(localArray.data(), localArray.size(), MPI_DOUBLE, finalArray.data(), localArray.size(), MPI_DOUBLE, root, MPI_COMM_WORLD);

    // Output the final temperatures at the specified points
    if (rank == root) {
        for (int i = 0; i < num_points; ++i) {
            int index = static_cast<int>(points[i] / h);
            output[i] = finalArray[index];
        }
    }

    MPI_Finalize();

    return;
}

void task_1() {
    // std::cout << std::endl << "Task #1: Algorithm implementation and testing" << std::endl;

    const double tau = 0.0002;
    const double t = 0.1;
    const double h = 0.02;

    // Generate points on the segment [0, 1] with interval 0.1
    const double output_point_dist = 0.1;
    int output_size = static_cast<int>(1 / output_point_dist) + 1;
    double points[output_size];
    for (int i = 0; i < output_size; i++) {
        points[i] = i * output_point_dist; 
    }

    double output[output_size]; // Array to store the output temperatures

    // Call the computation function
    tempDistribution(tau, t, h, points, output_size, output);

    // Print the output table header
    std::cout << std::setw(25) << "Point" << " | " << std::setw(25) << "Temperature" << std::endl;
    // Print the output temperatures
    for (int i = 0; i < output_size; i++) {
        std::cout << std::setw(25) << points[i] << " | " << std::setw(25) << output[i] << std::endl;
    }

    return;
}


int main() {
    task_1();   

    return 0;
}
