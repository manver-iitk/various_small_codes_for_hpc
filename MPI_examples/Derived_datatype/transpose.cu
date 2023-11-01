#include <iostream>
#include <mpi.h>

#define ROWS 3
#define COLS 4

void printArray(int* array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << array[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int array[ROWS][COLS] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    // Create the MPI transposed data type
    MPI_Datatype origType, transposedType;
    MPI_Type_vector(ROWS, 1, COLS, MPI_INT, &origType);
    MPI_Type_create_hvector(COLS, 1, sizeof(int), origType, &transposedType);
    MPI_Type_commit(&transposedType);

    if (rank == 0) {
        // Send the transposed array
        MPI_Send(&array[0][0], 1, transposedType, 1, 0, MPI_COMM_WORLD);
        std::cout << "Sent transposed array from rank 0" << std::endl;
    } else if (rank == 1) {
        // Receive the transposed array
        int receivedArray[COLS][ROWS];
        MPI_Recv(&receivedArray[0][0], COLS*ROWS, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Received transposed array at rank 1" << std::endl;
        printArray(&receivedArray[0][0], COLS, ROWS);
    }

    MPI_Type_free(&transposedType);
    MPI_Finalize();

    return 0;
}
