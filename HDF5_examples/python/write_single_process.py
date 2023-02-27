import numpy as np
import h5py as hp

Path_of_file = "/mnt/d/Lab_work/compairing_cuda_python/input"
File_name = "trash.h5"


Vkx = np.ones([1024,90,56],dtype=np.complex128)
Vkz = np.ones([1024,90,56],dtype=np.complex128)
Bkx = np.ones([1024,90,56],dtype=np.complex128)
Bkz = np.ones([1024,90,56],dtype=np.complex128)

File_handle = hp.File(Path_of_file + "/" + File_name,'w')
File_handle.create_dataset("Vkx",data=Vkx)
File_handle.create_dataset("Vkz",data=Vkz)
File_handle.create_dataset("Bkx",data=Bkx)
File_handle.create_dataset("Bkz",data=Bkz)

File_handle.close()

print(Vkx.shape)