#include "Finite.cuh"

template <typename T>
__global__ void finite_difference(T *input, T *output)
{
    long long int i = threadIdx.x + (blockIdx.x * gridDim.x);

    if (i >= (Nx_gpu * Ny_gpu * Nz_gpu))
    {
        return;
    }

    long long int z = i % Nz_gpu;
    long long int y = (i / Nz_gpu) % Ny_gpu;
    long long int x = (i / (Nz_gpu * Ny_gpu)) % Nx_gpu;

    __shared__ T data[256];
    data[threadIdx.x] = input[i];
    __syncthreads();

    if (axis == 0)
    {
        if ((z > 0) && (z < (Nz_gpu - 1)) && (threadIdx.x != 255) && (threadIdx.x != 0))
        {
            output[i] = (data[threadIdx.x + 1] - data[threadIdx.x - 1]) / 2.00;
        }
    }
}

template <typename T>
void init_arrays_memory()
{
    cudaMalloc(&(GPU_input_array<T>), sizeof(T) * Grid_dimension);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(GPU_output_array<T>), sizeof(T) * Grid_dimension);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    CPU_input_array<T> = (T *)malloc(sizeof(T) * Grid_dimension);
    CPU_output_array<T> = (T *)malloc(sizeof(T) * Grid_dimension);
}

template <typename T>
void fill_cpu_memory()
{
    for (long long int i = 0; i < (Grid_dimension); i++)
    {
        CPU_input_array<T>[i] = i;
    }
}

template <typename T>
void show_data(T *data)
{
    // std::cout << "\n Input data is :  " << std::endl;

    for (long long int i = 0; i < Nx; i++)
    {
        std::cout << "\n\n";
        for (long long int j = 0; j < Ny; j++)
        {
            std::cout << "\n";
            for (long long int k = 0; k < Nz; k++)
            {
                std::cout << "   " << data[(i * Ny * Nz) + (j * Nz) + k];
            }
        }
    }
    std::cout << "\n\n";
}

template <typename T>
void copy_data_between_devices(int direction)
{
    if (direction == 0)
    {
        cudaMemcpy(GPU_input_array<T>, CPU_input_array<T>, sizeof(T) * Grid_dimension, cudaMemcpyHostToDevice);
        gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
    }

    else if (direction == 1)
    {
        cudaMemcpy(CPU_output_array<T>, GPU_output_array<T>, sizeof(T) * Grid_dimension, cudaMemcpyDeviceToHost);
        gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
    }
}

template <typename T>
void calls_to_program()
{
    init_arrays_memory<T>();
    fill_cpu_memory<T>();
    show_data(CPU_input_array<T>);
    copy_data_between_devices<T>(0);

    // Calling of kernel
    finite_difference<<<{(Nx * Ny * Nz / 256) + 1, 1, 1}, {256, 1, 1}>>>(GPU_input_array<T>, GPU_output_array<T>);

    cudaDeviceSynchronize();
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    copy_data_between_devices<T>(1);
    show_data(CPU_output_array<T>);
}

int main(int argc, char **argv)
{

    // Setting the grid dimensions
    Nx = atoi(argv[1]);
    Nx = atoi(argv[2]);
    Nx = atoi(argv[3]);

    std::cout << "\n Nx = " << Nx;
    std::cout << "\n Ny = " << Ny;
    std::cout << "\n Nz = " << Nz;
    Grid_dimension = Nx * Ny * Nz;

    // Setting the dimensions to the GPU
    cudaMemcpyToSymbol(Nx_gpu, &Nx, sizeof(long long int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Ny_gpu, &Ny, sizeof(long long int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Nz_gpu, &Nz, sizeof(long long int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    // Start of code
    calls_to_program<double>();

    return 0;
}