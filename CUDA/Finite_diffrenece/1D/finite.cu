#include "Finite.cuh"

dim3 grid;
dim3 block;

template <typename T>
__global__ void finite_difference(T *input, T *output)
{
    long long int index = threadIdx.x + (blockIdx.x * gridDim.x);

    if (index >= N_gpu)
        return;

    T Front_data;
    T Back_data;
    int spacing;

    if (index == 0)
    {
        // Do the forward difference
        Front_data = input[index + 1];
        Back_data = input[index];
        spacing = 1;
    }
    if (index == (N_gpu - 1))
    {
        // Do the backward difference
        Front_data = input[index];
        Back_data = input[index - 1];
        spacing = 1;
    }
    if ((index > 0) && (index < (N_gpu - 1)))
    {
        Front_data = input[index + 1];
        Back_data = input[index - 1];
        spacing = 2;
    }
    __syncthreads();

    // Diffrenece scheme Formulae
    output[index] = (Front_data - Back_data) / spacing;
}

template <typename T>
void init_arrays_memory()
{
    cudaMalloc(&(GPU_input_array<T>), sizeof(T) * N);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(GPU_output_array<T>), sizeof(T) * N);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    CPU_input_array<T> = (T *)malloc(sizeof(T) * N);
    CPU_output_array<T> = (T *)malloc(sizeof(T) * N);
}

template <typename T>
void fill_cpu_memory()
{
    for (long long int i = 0; i < N; i++)
    {
        CPU_input_array<T>[i] = i;
    }
}

template <typename T>
void show_data(T *data, int in_out)
{
    if (in_out == 0)
    {
        std::cout << "\n Input data is :  " << std::endl;
    }
    else
    {
        std::cout << "\n Output data is :  " << std::endl;
    }

    for (long long int i = 0; i < N; i++)
    {
        std::cout << data[i] << ",";
    }

    std::cout << "\n\n";
}

template <typename T>
void copy_data_between_devices(int direction)
{
    if (direction == 0)
    {
        cudaMemcpy(GPU_input_array<T>, CPU_input_array<T>, sizeof(T) * N, cudaMemcpyHostToDevice);
        gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
    }

    else if (direction == 1)
    {
        cudaMemcpy(CPU_output_array<T>, GPU_output_array<T>, sizeof(T) * N, cudaMemcpyDeviceToHost);
        gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
    }
}

int main(int argc, char **argv)
{

    // Setting the grid dimensions
    N = atoi(argv[1]);

    std::cout << "\n N = " << N;

    // Setting the grid and block
    grid = {((N / 256) + 1), 1, 1};
    block = {256, 1, 1};

    // Setting the dimensions to the GPU
    cudaMemcpyToSymbol(N_gpu, &N, sizeof(long long int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    // Allocate the Memoery in CPU
    init_arrays_memory<double>();

    // Initialize the Memoery in CPU
    fill_cpu_memory<double>();

    // Show the data in CPU
    show_data(CPU_input_array<double>, 0);

    // Copy Data from CPU to GPU
    copy_data_between_devices<double>(0);

    // Calling of kernel
    finite_difference<<<grid, block, 0, 0>>>(GPU_input_array<double>, GPU_output_array<double>);

    cudaDeviceSynchronize();
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    // Copy Data from GPU to CPU
    copy_data_between_devices<double>(1);

    // Show the data after compuatation
    show_data(CPU_output_array<double>, 1);

    return 0;
}