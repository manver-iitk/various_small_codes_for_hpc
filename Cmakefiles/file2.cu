#pragma once
#include "header.cuh"

int Nx{512};
int Ny{512};
int Nz{512};

void set_para()
{
    Nx = 256;
    Ny = 256;
    Nz = 256;

    std::cout << "\n Nx = " << Nx
              << "\n Ny = " << Ny
              << "\n Nz = " << Nz << std::endl;
}