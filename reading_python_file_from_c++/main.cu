#include <iostream>
#include <Python.h>
#include <cuda.h>
#include <cufft.h>
#include <complex.h>

PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

std::complex<double> data1;
Py_complex data;

__forceinline__ void python_err_check(PyObject *data_pointer, int line, std::string file_name) // Python Error Checker
{
    if (data_pointer == NULL)
    {
        std::cout << "Error in Python Functions call at line no " << line << "\n In File " << file_name << " , aborting " << std::endl;
        exit(0);
    }
}

int main(int argc, char **argv)
{

    // ############ initialize the Python reading #########
    Py_Initialize();

    PyRun_SimpleString("import sys\n"
                       "import os\n"
                       "sys.path.append(os.getcwd())\n");
    // ####################################################

    // Opening Python Script file
    pName = PyUnicode_FromString((char *)"para");
    python_err_check(pName, __LINE__, __FILE__);
    pModule = PyImport_Import(pName);
    python_err_check(pModule, __LINE__, __FILE__);

    // Opening The function named return_data2_type in python script
    pFunc = PyObject_GetAttrString(pModule, (char *)"return_data2_type");
    python_err_check(pFunc, __LINE__, __FILE__);
    pValue = PyObject_CallObject(pFunc, nullptr);
    python_err_check(pValue, __LINE__, __FILE__);

    // reading the complex datatype from the python file 
    data = (PyComplex_AsCComplex(PyList_GET_ITEM(pValue, 0)));
    std::cout << data.imag;
    std::cout << data.real;
    std::cout << std::endl;
    data1 = std::complex<double>(data.real, data.imag);
    std::cout << data1;
    std::cout << std::endl;

    Py_Finalize();
    return 0;
}