#include "header.cuh"


__global__ void init_curand_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(clock64(), id, 0, &state[0]);

}


__global__ void generate_kernel(curandState *state,
                                int n,
                                double *result, double *result_2)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;

    /* Copy state to local memory for efficiency */
    curandState localState;

    curand_init(clock64(), id, 0, &localState);

    result[id] = curand_uniform_double(&localState);
    result_2[id] = curand_uniform_double(&localState);
    // result[id] += count;
}

int main()
{

    // Initialize the curand on device
    data_cpu = (double *)calloc(total_threads, sizeof(double));
    data_cpu_2 = (double *)calloc(total_threads, sizeof(double));
    cudaMalloc(&data_gpu, sizeof(double) * total_threads);
    cudaMalloc(&data_gpu_2, sizeof(double) * total_threads);
    cudaMemset(data_gpu, 0, sizeof(double) * total_threads);
    cudaMemset(data_gpu_2, 0, sizeof(double) * total_threads);

    // cudaMalloc((void **)&devMRGStates, total_threads * sizeof(curandStateMRG32k3a));
    // cudaMalloc((void **)&devPHILOXStates, total_threads * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((void **)&devStates, total_threads * sizeof(curandState));

    std::cout << "\n data size = " << sizeof(curandState);
    std::cout << "\n data size int = " << sizeof(int );

    /* Generate and use pseudo-random  */
    // init_curand_kernel<<<grid_dim, block_dim>>>(devStates);

    generate_kernel<<<grid_dim, block_dim>>>(devStates, sampleCount, data_gpu, data_gpu_2);

    cudaMemcpy(data_cpu, data_gpu, total_threads * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_cpu_2, data_gpu_2, total_threads * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    for (size_t i = 0; i < total_threads; i++)
    {
        std::cout << "  " << data_cpu[i];
        if (i % 10 == 0)
        {
            std::cout << "\n";
        }
    }
    std::cout << std::endl;

    for (size_t i = 0; i < total_threads; i++)
    {
        std::cout << "  " << data_cpu_2[i];
        if (i % 10 == 0)
        {
            std::cout << "\n";
        }
    }

    std::cout << std::endl;

    return 0;
}