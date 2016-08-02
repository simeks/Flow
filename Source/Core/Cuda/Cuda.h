#ifndef __CUDA_CUDA_H__
#define __CUDA_CUDA_H__

#include <cuda_runtime.h>


template<typename T>
void cuda_check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        FATAL_ERROR("CUDA error at %s:%d \"%s\" : %s \n",
            file, line, static_cast<unsigned int>(result), func, cudaGetErrorString(result));
    }
}

#define CUDA_CHECK_ERRORS(val) cuda_check((val), #val, __FILE__, __LINE__)

#endif // __CUDA_CUDA_H__
