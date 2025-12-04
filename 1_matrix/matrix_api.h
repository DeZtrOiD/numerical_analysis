#pragma once
#ifdef BUILD_DLL
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT __declspec(dllimport)
#endif

#include "matrix.h"

extern "C" {

DLL_EXPORT void* __cdecl create_matrix(int rows, int cols, double* data);
DLL_EXPORT void __cdecl free_matrix(void* matrix_ptr);

DLL_EXPORT void __cdecl matrix_solve_thomas(void* matrix_ptr, double* b, double* result);
DLL_EXPORT void __cdecl matrix_print(void* matrix_ptr);
DLL_EXPORT void __cdecl matrix_solve_gauss(void* matrix_ptr, double* b, double* result);

}
