#include "Common.h"
#include "Cuda.h"
#include "CudaImage.h"

#include "Image/Image.h"

CudaImage::CudaData::CudaData(int ndims, const Vec3i& size, size_t elem_size)
{
    if (ndims == 1)
    {
        CUDA_CHECK_ERRORS(cudaMalloc(&data, elem_size * size.x));
        step = elem_size * size.x;
    }
    else if (ndims == 2)
    {
        CUDA_CHECK_ERRORS(cudaMallocPitch(&data, &step, elem_size * size.x, size.y));
    }
    else
    {
        cudaPitchedPtr ptr;
        cudaExtent extent = make_cudaExtent(elem_size * size.x, size.y, size.z);
        CUDA_CHECK_ERRORS(cudaMalloc3D(&ptr, extent));

        data = (uint8_t*)ptr.ptr;
        step = ptr.pitch;
    }
}
CudaImage::CudaData::~CudaData()
{
    cudaFree(data);
    data = nullptr;
}

CudaImage::CudaImage() :
    _data(nullptr),
    _ndims(0),
    _pixel_type(image::PixelType_Unknown)
{
}
CudaImage::CudaImage(int ndims, const Vec3i& size, int type)
{
    create(ndims, size, type);
}
CudaImage::~CudaImage()
{
    _data.reset();
}

void CudaImage::create(int ndims, const Vec3i& size, int type)
{
    assert(ndims <= 3);
    assert(type != image::PixelType_Unknown);

    release();

    _ndims = ndims;
    _size = size;
    _pixel_type = type;

    size_t elem_size = image::pixel_size(type);
    _data = std::make_shared<CudaData>(ndims, size, elem_size);
}
void CudaImage::release()
{
    _pixel_type = image::PixelType_Unknown;
    _data.reset();
}

CudaImage::CudaImage(const CudaImage& other) :
    _data(other._data),
    _ndims(other._ndims),
    _size(other._size),
    _pixel_type(other._pixel_type)
{
}
CudaImage& CudaImage::operator=(const CudaImage& other)
{
    _data = other._data;
    _ndims = other._ndims;
    _size = other._size;
    _pixel_type = other._pixel_type;

    return *this;
}
