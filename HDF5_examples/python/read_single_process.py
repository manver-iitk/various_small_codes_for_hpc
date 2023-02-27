import numpy as np
import h5py as hp

Path_of_file = "/mnt/d/Lab_work/compairing_cuda_python/input"
File_name = "init_cond_random_1024.h5"


File_handle = hp.File(Path_of_file + "/" + File_name,'r')
Vkx = np.asarray(File_handle["Vkx"])
Vkz = np.asarray(File_handle["Vkz"])
Bkx = np.asarray(File_handle["Bkx"])
Bkz = np.asarray(File_handle["Bkz"])

File_handle.close()

print(Vkx.shape)
