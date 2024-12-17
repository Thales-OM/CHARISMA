# Define the compiler and flags
CC = gcc
CFLAGS = -lstdc++ -fopenmp -lopenblas

# Define the executable name
EXECUTABLE = gemm_pugin_1

# Define the source file
SOURCE = gemm_pugin_1.cpp

# Define the modules to load
MODULES = openmpi/5.0.3 INTEL/oneAPI_2021_u2 mkl/latest OpenBlas/v0.3.18

# Define the rule to compile and run the program
all:
	module load  $(MODULES)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(SOURCE)
	srun -n 1 -c 48 --constraint="type_d" --time=00:10:00 ./$(EXECUTABLE)

# Define the rule to clean the output files
clean:
	rm -f $(EXECUTABLE) gemm_pugin_1*