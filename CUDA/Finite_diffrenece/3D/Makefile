NVCC= nvcc
CXX_STANDARD= -std=c++17
compute_arch= -arch=sm_75

File_name= finite
File_extension= cu

File_name_header= Finite
File_extension_header= cuh

compile: $(File_name).$(File_extension) $(File_name_header).$(File_extension_header)
	$(NVCC) $(CXX_STANDARD) $(compute_arch) $(File_name).$(File_extension) -o $(File_name)

run: $(File_name)
	./$(File_name)

clean: 
	rm $(File_name)