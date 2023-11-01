#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

long long int Nx;
long long int Ny;
long long int Nz;

__constant__ long long int Nx_gpu;
__constant__ long long int Ny_gpu;
__constant__ long long int Nz_gpu;

long long int Grid_dimension{0};


template<typename T>
T* CPU_input_array;

template<typename T>
T* GPU_input_array;

template<typename T>
T* GPU_output_array;

template<typename T>
T* CPU_output_array;

void gpuerrcheck_cudaerror(cudaError_t err, long long int line, std::string file_name) // CUDA ERROR CHECKER
{
    if (err != 0)
    {
        std::cout << "\n cuda error  = " << cudaGetErrorString(err) << " , At line " << line << "\n In File " << file_name << " , aborting " << std::endl;
        exit(0);
    }
}