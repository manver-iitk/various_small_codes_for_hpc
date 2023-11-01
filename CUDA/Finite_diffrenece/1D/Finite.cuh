#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

long long int N;
__constant__ long long int N_gpu;


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