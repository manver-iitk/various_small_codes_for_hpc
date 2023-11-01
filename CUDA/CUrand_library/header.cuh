#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <curand.h>
#include <curand_kernel.h>

curandState *devStates;
curandStateMRG32k3a *devMRGStates;
curandStatePhilox4_32_10_t *devPHILOXStates;

bool useMRG = 0;
bool usePHILOX = 0;
int sampleCount = 10000;
bool doubleSupported = 0;

double *data_gpu, *data_gpu_2;
double *data_cpu, *data_cpu_2;

int block_dim = 10;
int grid_dim = 5;
int total_threads{block_dim * grid_dim};
