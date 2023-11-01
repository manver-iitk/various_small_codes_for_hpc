#pragma once
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern int Nx;
extern int Ny;
extern int Nz;

extern "C++" __global__ void test_kernel();

extern "C++" void set_para();