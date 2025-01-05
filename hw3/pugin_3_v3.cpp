#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>


void tempDistributionAsync(double tau, double t, double h, double* points, int num_points, double* output) {
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

    // Prepare to receive neighbor values
    double leftNeighbor = (startIndex > 0) ? 1.0 : 0.0; // Left neighbor
    double rightNeighbor = (endIndex < arraySize) ? 1.0 : 0.0; // Right neighbor

    MPI_Request requests[4]; // Array to hold requests for non-blocking operations

    // Asynchronous send and receive operations
    if (rank > 0) {
        MPI_Isend(&localArray[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]); // Send leftmost value
    }

    if (rank < size - 1) {
        MPI_Irecv(&rightNeighbor, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[1]); // Receive right neighbor
    }

    if (rank < size - 1) {
        MPI_Isend(&localArray.back(), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[2]); // Send rightmost value
    }

    if (rank > 0) {
        MPI_Irecv(&leftNeighbor, 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[3]); // Receive left neighbor
    }

    // Main time-stepping loop
    for (int step = 0; step < totalSteps; ++step) {
        std::vector<double> nextArray(localArray.size(), 0.0);

        // Flags for received neighbors
        bool leftReceived = false;
        bool rightReceived = false;
        bool leftPlusOneComputed = false;
        bool rightMinusOneComputed = false;
        bool leftSent = false;
        bool rightSent = false;

        int current_middle_idx = 2; // start from left + 2 for middle computation
        int max_middle_idx = (nextArray.size() > 4) ? -1 : nextArray.size() - 3; // max middle index to compute
        
        // Main computation loop
        while ((current_middle_idx <= max_middle_idx) || !leftSent || !rightSent) {
            if (!leftPlusOneComputed && nextArray.size() > 2) {
                nextArray[1] = (localArray[0] + localArray[2]) / 2.0;
                leftPlusOneComputed = true;
            }

            // Check if left neighbor has been received
            if (rank > 0 && !leftReceived) {
                MPI_Status status;
                int flag;
                MPI_Test(&requests[3], &flag, &status); // Check left neighbor
                if (flag) {
                    leftReceived = true; // Mark left neighbor as received
                }

            }

            // Compute leftmost element if left neighbor is received
            if (leftReceived && leftPlusOneComputed && (nextArray.size() >= 2) && !leftSent) {
                nextArray[0] = (leftNeighbor + localArray[1]) / 2.0; // Compute using left neighbor

                // Send the new leftmost value to the left neighbor
                if (rank > 0) {
                    MPI_Isend(&nextArray[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
                    leftSent = true;
                }

                if (nextArray.size() == 2) {
                    rightMinusOneComputed = true;
                }
            }

            if (leftReceived && rightReceived && (nextArray.size() == 1) && !leftSent) {
                nextArray[0] = (leftNeighbor + rightNeighbor) / 2.0; // Compute using left neighbor

                // Send the new leftmost value to the left neighbor
                if (rank > 0) {
                    MPI_Isend(&nextArray[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
                    leftSent = true;
                }
            }

            // Compute rightmost element if right neighbor is received
            if (!rightMinusOneComputed && nextArray.size() > 2) {
                nextArray[localArray.size() - 2] = (localArray[localArray.size() - 1] + localArray[localArray.size() - 3]) / 2.0;
                rightMinusOneComputed = true;
            }

            // Check if right neighbor has been received
            if (rank < size - 1 && !rightReceived) {
                MPI_Status status;
                int flag;
                MPI_Test(&requests[1], &flag, &status); // Check right neighbor 
                if (flag) {
                    rightReceived = true; // Mark right neighbor as received
                }
                if (localArray.size() == 2) {
                    leftPlusOneComputed = true;
                }
            }

            // Compute rightmost element if right neighbor is received
            if (rightReceived && rightMinusOneComputed && (nextArray.size() >= 2) && !rightSent) {
                nextArray.back() = (localArray[localArray.size() - 2] + rightNeighbor) / 2.0; // Compute using right neighbor

                // Send the new rightmost value to the right neighbor
                if (rank < size - 1) {
                    MPI_Isend(&nextArray.back(), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[2]);
                    rightSent = true;
                }
            }

            if (rightReceived && leftReceived && (nextArray.size() ==1) && !rightSent) {
                nextArray.back() = (leftNeighbor + rightNeighbor) / 2.0; // Compute using right neighbor

                // Send the new rightmost value to the right neighbor
                if (rank < size - 1) {
                    MPI_Isend(&nextArray.back(), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[2]);
                    rightSent = true;
                }
            }

            if (current_middle_idx <= max_middle_idx) {
                nextArray[current_middle_idx] = (localArray[current_middle_idx - 1] + localArray[current_middle_idx + 1]) / 2.0; // Compute using local values
                current_middle_idx++;
            }
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
}

void task_3() {
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
    tempDistributionAsync(tau, t, h, points, output_size, output);

    // Print the output table header
    std::cout << std::setw(25) << "Point" << " | " << std::setw(25) << "Temperature" << std::endl;
    // Print the output temperatures
    for (int i = 0; i < output_size; i++) {
        std::cout << std::setw(25) << points[i] << " | " << std::setw(25) << output[i] << std::endl;
    }

    return;
}

int main() {   
    task_3();
    
    return 0;
}