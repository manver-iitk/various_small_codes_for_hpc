import numpy as np
import matplotlib.pyplot as plt

Grid_size = np.asarray([64*64*64, 64*64*128, 64*128*128, 128*128*128, 128*128 *
                        256, 128*256*256, 256*256*256, 256*256*512, 256*512*512, 512*512*512])
CPU_time_numpy = np.asarray([0.495454545, 0.8825, 2.725,
                             5.2425, 16.525, 38, 85.5, 224.5, 533.35, 1378.65])

GPU_time_CUDA_A100 = np.asarray([0.002, 0.0034, 0.005, 0.01, 0.018, 0.036, 0.074, 0.14, 0.27, 0.57
                                 ])
GPU_time_Python_A100 = np.asarray([0.1, 0.133636364, 0.172727273, 0.213636364, 0.336363636, 0.590909091, 1.109090909, 2.5, 6.068181818, 12.17272727
                                   ])

GPU_time_CUDA_V100 = np.asarray(
    [0.003, 0.0056, 0.011, 0.025, 0.041, 0.081, 0.163, 0.318, 0.664])
GPU_time_Python_V100 = np.asarray([0.11, 0.15, 0.2, 0.227272727, 0.363636364, 0.663636364, 1.254545455, 3.070909091
                                   ])

GPU_time_CUDA_TitanX = np.asarray([0.008, 0.017, 0.031, 0.06, 0.119, 0.231, 0.458, 0.927
                                   ])
GPU_time_Python_TitanX = np.asarray(
    [0.171188119, 0.272727273, 0.354545455, 0.490909091, 0.749181818, 1.472727273, 2.909090909, 7.290909091])

GPU_time_CUDA_RTX_2060_laptop_grade = np.asarray([0.015, 0.027, 0.051, 0.099, 0.19, 0.376, 0.744
                                                  ])
GPU_time_Python_RTX_2060_laptop_grade = np.asarray(
    [0.898181818, 0.988181818, 1.062727273, 1.197272727, 1.768181818, 2.838818182, 5.100909091])

GPU_time_CUDA_K40 = np.asarray([0.027, 0.033, 0.066, 0.134, 0.216, 0.436, 0.875, 1.683
                                ])
GPU_time_Python_K40 = np.asarray([0.245454545, 0.354545455, 0.456363636, 0.582727273, 1.214545455, 2.596363636, 5.690909091, 14.90909091
                                  ])

GPU_time_CUDA_K10 = np.asarray([0.035, 0.054, 0.113, 0.229, 0.439, 0.903, 1.843
                                ])
GPU_time_Python_K10 = np.asarray(
    [0.302727273, 0.327272727, 0.490909091, 0.89, 2.018181818, 5.283636364])

GPU_time_CUDA_RTX_3050_laptop_grade = np.asarray(
    [0.016, 0.035, 0.064, 0.13, 0.258, 0.516, 1.021])
GPU_time_Python_RTX_3050_laptop_grade = np.asarray(
    [0.131818182, 0.201818182, 0.327272727, 0.563636364, 1.209090909, 2.609090909
     ])

GPU_time_CUDA_RTX_4090_desktop_grade = np.asarray(
    [0.002, 0.0045, 0.0083, 0.016, 0.031, 0.062, 0.123, 0.248, 0.5
     ])
GPU_time_Python_RTX_4090_desktop_grade = np.asarray(
    [0.068181818, 0.1, 0.118181818, 0.154545455, 0.245454545, 0.514545455, 1.045454545, 2.545454545, 5.845454545])

plot_speedups_for = "CUDA"
start_index = 1

if (plot_speedups_for == "CUDA"):

    Speedup_CUDA_numpy_python_A100 = (
        CPU_time_numpy[0:GPU_time_CUDA_A100.shape[0]] / GPU_time_CUDA_A100[:])
    Speedup_CUDA_numpy_python_V100 = (
        CPU_time_numpy[0:GPU_time_CUDA_V100.shape[0]] / GPU_time_CUDA_V100[:])
    Speedup_CUDA_numpy_python_TitanX = (
        CPU_time_numpy[0:GPU_time_CUDA_TitanX.shape[0]] / GPU_time_CUDA_TitanX[:])
    Speedup_CUDA_numpy_python_RTX_2060_laptop_grade = (
        CPU_time_numpy[0:GPU_time_CUDA_RTX_2060_laptop_grade.shape[0]] / GPU_time_CUDA_RTX_2060_laptop_grade[:])
    Speedup_CUDA_numpy_python_RTX_3050_laptop_grade = (
        CPU_time_numpy[0:GPU_time_CUDA_RTX_3050_laptop_grade.shape[0]] / GPU_time_CUDA_RTX_3050_laptop_grade[:])
    Speedup_CUDA_numpy_python_RTX_4090_desktop_grade = (
        CPU_time_numpy[0:GPU_time_CUDA_RTX_4090_desktop_grade.shape[0]] / GPU_time_CUDA_RTX_4090_desktop_grade[:])
    Speedup_CUDA_numpy_python_K40 = (
        CPU_time_numpy[0:GPU_time_CUDA_K40.shape[0]] / GPU_time_CUDA_K40[:])
    Speedup_CUDA_numpy_python_K10 = (
        CPU_time_numpy[0:GPU_time_CUDA_K10.shape[0]] / GPU_time_CUDA_K10[:])

    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_A100.shape[0]],
               Speedup_CUDA_numpy_python_A100[start_index:], label="A100", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_A100.shape[0]-1],Speedup_CUDA_numpy_python_A100[-1]," A100_40_GB")
        
    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_V100.shape[0]],
               Speedup_CUDA_numpy_python_V100[start_index:], label="V100", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_V100.shape[0]-1],Speedup_CUDA_numpy_python_V100[-1]," V100_16_GB")
    
    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_TitanX.shape[0]],
               Speedup_CUDA_numpy_python_TitanX[start_index:], label="TitanX", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_TitanX.shape[0]-1],Speedup_CUDA_numpy_python_TitanX[-1]," TiTanX_12_GB")
    
    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_RTX_2060_laptop_grade.shape[0]],
               Speedup_CUDA_numpy_python_RTX_2060_laptop_grade[start_index:], label="RTX_2060 (LG)", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_RTX_2060_laptop_grade.shape[0]-1],Speedup_CUDA_numpy_python_RTX_2060_laptop_grade[-1]," RTX_2060_LG_6GB")
    
    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_RTX_3050_laptop_grade.shape[0]],
               Speedup_CUDA_numpy_python_RTX_3050_laptop_grade[start_index:], label="RTX_3050 (LG)", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_RTX_3050_laptop_grade.shape[0]-1],Speedup_CUDA_numpy_python_RTX_3050_laptop_grade[-1]," RTX_3050_LG_4GB")

    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_RTX_4090_desktop_grade.shape[0]],
               Speedup_CUDA_numpy_python_RTX_4090_desktop_grade[start_index:], label="RTX_4090 (DG)", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_RTX_4090_desktop_grade.shape[0]-1],Speedup_CUDA_numpy_python_RTX_4090_desktop_grade[-1]," RTX_4090_DG_25GB")
    
    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_K40.shape[0]],
               Speedup_CUDA_numpy_python_K40[start_index:], label="K40", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_K40.shape[0]-1],Speedup_CUDA_numpy_python_K40[-1]," K40_12_GB")
    
    plt.loglog(Grid_size[start_index:Speedup_CUDA_numpy_python_K10.shape[0]],
               Speedup_CUDA_numpy_python_K10[start_index:], label="K10", marker='^')
    # plt.text(Grid_size[Speedup_CUDA_numpy_python_K10.shape[0]-1],Speedup_CUDA_numpy_python_K10[-1]," K10_4_GB")

    plt.title(" Speedups comparison with single core cpu For CUDA(C/C++) ")
    plt.xlabel("Grid Size")
    plt.ylabel("Speedup")
    plt.legend()
    plt.tight_layout()
    plt.savefig("CUDA_speedup_graph")
    # plt.show()

if (plot_speedups_for == "Python"):

    Speedup_Python_numpy_python_A100 = (
        CPU_time_numpy[0:GPU_time_Python_A100.shape[0]] / GPU_time_Python_A100[:])
    Speedup_Python_numpy_python_V100 = (
        CPU_time_numpy[0:GPU_time_Python_V100.shape[0]] / GPU_time_Python_V100[:])
    Speedup_Python_numpy_python_V100 = (
        CPU_time_numpy[0:GPU_time_Python_V100.shape[0]] / GPU_time_Python_V100[:])
    Speedup_Python_numpy_python_TitanX = (
        CPU_time_numpy[0:GPU_time_Python_TitanX.shape[0]] / GPU_time_Python_TitanX[:])
    Speedup_Python_numpy_python_RTX_2060_laptop_grade = (
        CPU_time_numpy[0:GPU_time_Python_RTX_2060_laptop_grade.shape[0]] / GPU_time_Python_RTX_2060_laptop_grade[:])
    Speedup_Python_numpy_python_RTX_3050_laptop_grade = (
        CPU_time_numpy[0:GPU_time_Python_RTX_3050_laptop_grade.shape[0]] / GPU_time_Python_RTX_3050_laptop_grade[:])
    Speedup_Python_numpy_python_RTX_4090_desktop_grade = (
        CPU_time_numpy[0:GPU_time_Python_RTX_4090_desktop_grade.shape[0]] / GPU_time_Python_RTX_4090_desktop_grade[:])
    Speedup_Python_numpy_python_K40 = (
        CPU_time_numpy[0:GPU_time_Python_K40.shape[0]] / GPU_time_Python_K40[:])
    Speedup_Python_numpy_python_K10 = (
        CPU_time_numpy[0:GPU_time_Python_K10.shape[0]] / GPU_time_Python_K10[:])

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_A100.shape[0]],
               Speedup_Python_numpy_python_A100[start_index:], label="A100", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_A100.shape[0]-1],Speedup_Python_numpy_python_A100[-1]," A100_40_GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_V100.shape[0]],
               Speedup_Python_numpy_python_V100[start_index:], label="V100", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_V100.shape[0]-1],Speedup_Python_numpy_python_V100[-1]," V100_16_GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_TitanX.shape[0]],
               Speedup_Python_numpy_python_TitanX[start_index:], label="TitanX", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_TitanX.shape[0]-1],Speedup_Python_numpy_python_TitanX[-1]," TiTanX_12_GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_RTX_2060_laptop_grade.shape[0]],
               Speedup_Python_numpy_python_RTX_2060_laptop_grade[start_index:], label="RTX_2060 (LG)", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_RTX_2060_laptop_grade.shape[0]-1],Speedup_Python_numpy_python_RTX_2060_laptop_grade[-1]," RTX_2060_LG_6GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_RTX_3050_laptop_grade.shape[0]],
               Speedup_Python_numpy_python_RTX_3050_laptop_grade[start_index:], label="RTX_3050 (LG)", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_RTX_3050_laptop_grade.shape[0]-1],Speedup_Python_numpy_python_RTX_3050_laptop_grade[-1]," RTX_3050_LG_4GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_RTX_4090_desktop_grade.shape[0]],
               Speedup_Python_numpy_python_RTX_4090_desktop_grade[start_index:], label="RTX_4090 (DG)", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_RTX_4090_desktop_grade.shape[0]-1],Speedup_Python_numpy_python_RTX_4090_desktop_grade[-1]," RTX_4090_DG_25GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_K40.shape[0]],
               Speedup_Python_numpy_python_K40[start_index:], label="K40", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_K40.shape[0]-1],Speedup_Python_numpy_python_K40[-1]," K40_12_GB")

    plt.loglog(Grid_size[start_index:Speedup_Python_numpy_python_K10.shape[0]],
               Speedup_Python_numpy_python_K10[start_index:], label="K10", marker='^')
    # plt.text(Grid_size[Speedup_Python_numpy_python_K10.shape[0]-1],Speedup_Python_numpy_python_K10[-1]," K10_4_GB")

    plt.title(" Speedups comparison with single core cpu For cupy(Python) ")
    plt.xlabel("Grid Size")
    plt.ylabel("Speedup")
    plt.legend()
    plt.tight_layout()
    plt.savefig("CUPY_speedup_graph")
    # plt.show()
