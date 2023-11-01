#pragma once
#include "header.cuh"

__global__ void test_kernel()
{
    for (size_t i = 0; i < 10; i++)
    {
        printf("\n hello this is id ---> %d", i);
    }
}

int main()
{
    cudaDeviceSynchronize();
    test_kernel<<<1, 1, 0, 0>>>();
    cudaDeviceSynchronize();

    set_para();

    return 0;
}