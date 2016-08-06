#ifndef __CUDA_CUDA_H__
#define __CUDA_CUDA_H__

#include <cuda_runtime.h>

template<typename T>
void cuda_check(T result, const char* func, const char* file, int line)
{
    if (result)
    {
        FATAL_ERROR("CUDA error at %s:%d \"%s\" : %s\n",
            file, line, func, cudaGetErrorString(result));
    }
}

INLINE void cuda_get_last_error(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        FATAL_ERROR("CUDA error at %s:%d : %s\n", file, line, cudaGetErrorString(err));
    }
}

#define CUDA_CHECK_ERRORS(val) cuda_check((val), #val, __FILE__, __LINE__)
#define CUDA_GET_LAST_ERROR() cuda_get_last_error(__FILE__, __LINE__)

#endif // __CUDA_CUDA_H__
