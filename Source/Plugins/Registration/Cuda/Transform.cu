#include <Core/Common.h>
#include <Core/Cuda/Cuda.h>
#include <Core/Cuda/CudaImage.h>

#include "Transform.h"

#include <cuda_runtime.h>

namespace cuda
{
    template<typename T> __device__ T read_value(const T* src, int x, int y, int z, const cudaExtent size)
    {
        return src[x + y * size.width + z * size.width * size.height];
    }
    template<typename T> __device__ void write_value(T* dst, int x, int y, int z, const cudaExtent size, T v)
    {
        dst[x + y * size.width + z * size.width * size.height] = v;
    }


    texture<float, 3> tex_transform_3d_float;
    __global__ void transform_image_3d_float_kernel(const double3* def, 
                                                    float* out,
                                                    cudaExtent size)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        int z = blockIdx.z*blockDim.z + threadIdx.z;

        if (x >= size.width || y >= size.height || z >= size.depth)
            return;
        
        double3 d = read_value<double3>(def, x, y, z, size);
        write_value<float>(out, x, y, z, size, tex3D(tex_transform_3d_float, x + d.x + 0.5f, y + d.y + 0.5f, z + d.z + 0.5f));
    }


    Image transform_image(const Image& source, const ImageVec3d& deformation)
    {
        assert(source.size() == deformation.size());

        Image result;

        CudaImage gpu_def(deformation);

        dim3 block(8, 8, 1);
        dim3 grid(source.size().x / block.x, source.size().y / block.y, source.size().z);

        if (source.pixel_type() == image::PixelType_Float32)
        {
            CudaImage gpu_source(source, CudaImage::Flag_Texture);
            CudaImage gpu_dest(source.ndims(), source.size(), image::PixelType_Float32);

            tex_transform_3d_float.addressMode[0] = cudaAddressModeBorder;
            tex_transform_3d_float.addressMode[1] = cudaAddressModeBorder;
            tex_transform_3d_float.filterMode = cudaFilterModeLinear;

            cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
            CUDA_CHECK_ERRORS(cudaBindTextureToArray(&tex_transform_3d_float, 
                gpu_source.cuda_array(), &desc));

            transform_image_3d_float_kernel << <grid, block >> >(gpu_def.ptr<double3>(), gpu_dest.ptr<float>(), gpu_source.cuda_extent());
            CUDA_GET_LAST_ERROR();

            gpu_dest.download(result);
        }
        else
        {
            FATAL_ERROR("Unsupported format for transformation.");
        }


        result.set_spacing(source.spacing());
        result.set_origin(source.origin());
        return result;
    }
}


