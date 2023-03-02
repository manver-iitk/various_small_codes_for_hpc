NVCC= nvcc -std=c++17 -arch=sm_75

CXX= g++ -std=c++14
CC= gcc -std=c14
python= python3

CUDA_HOME=/usr/local/cuda-12.0
CUDA_INC=$(CUDA_HOME)/include/
CUDA_LIB=$(CUDA_HOME)/lib64/

PYTHON_INC=/usr/include/python3.10/
PYTHON_LIB=/usr/lib/python3.10/config-3.10-x86_64-linux-gnu/

Python = -lpython3.10

Oflag= -Xptxas -O3

INC_ALL= $(CUDA_INC),$(PYTHON_INC)
LIB_ALL= $(CUDA_LIB),$(PYTHON_LIB)


Link_all= $(Python)


all: compile run_file

compile: main.cu
	$(NVCC) main.cu -I $(INC_ALL) -L $(LIB_ALL) $(NVCC_FLAGS) $(Oflag) $(Link_all) -o run


run_file: run
	LD_LIBRARY_PATH=$(PYTHON_LIB):$(LD_LIBRARY_PATH) ./run