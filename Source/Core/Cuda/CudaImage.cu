#include "Common.h"
#include "Cuda.h"
#include "CudaImage.h"

#include "Image/Image.h"

CudaImage::CudaData::CudaData(int ndims, const Vec3i& size, size_t elem_size)
{
    if (ndims == 1)
    {
        CUDA_CHECK_ERRORS(cudaMalloc(&data, elem_size * size.x));
        pitch = elem_size * size.x;
    }
    else if (ndims == 2)
    {
        CUDA_CHECK_ERRORS(cudaMallocPitch(&data, &pitch, elem_size * size.x, size.y));
    }
    else
    {
        cudaPitchedPtr ptr;
        cudaExtent extent = make_cudaExtent(elem_size * size.x, size.y, size.z);
        CUDA_CHECK_ERRORS(cudaMalloc3D(&ptr, extent));

        data = (uint8_t*)ptr.ptr;
        pitch = ptr.pitch;
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

    if (_ndims == ndims && _size == size && _pixel_type == type && _data)
        return;

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
void CudaImage::upload(const Image& img)
{
    create(img.ndims(), img.size(), img.pixel_type());

    size_t elem_size = image::pixel_size(_pixel_type);
    if (_ndims == 1)
    {
        CUDA_CHECK_ERRORS(cudaMemcpy(_data->data, img.ptr(), elem_size * _size.x, cudaMemcpyHostToDevice));
    }
    else if (_ndims == 2)
    {
        CUDA_CHECK_ERRORS(cudaMemcpy2D(_data->data, _data->pitch, img.ptr(), img.step()[1], elem_size * _size.x, _size.y, cudaMemcpyHostToDevice));
    }
    else
    {
        cudaMemcpy3DParms parms = { 0 };
        parms.dstPtr = make_cudaPitchedPtr(_data->data, _data->pitch, elem_size * _size.x, _size.y);
        parms.srcPtr = make_cudaPitchedPtr((void*)img.ptr(), img.step()[1], elem_size * img.size().x, img.size().y);
        parms.extent = make_cudaExtent(elem_size * _size.x, _size.y, _size.z);
        parms.kind = cudaMemcpyHostToDevice;
        
        CUDA_CHECK_ERRORS(cudaMemcpy3D(&parms));
    }
}
void CudaImage::download(Image& img)
{
    img.create(_ndims, _size, _pixel_type);

    size_t elem_size = image::pixel_size(_pixel_type);
    if (_ndims == 1)
    {
        CUDA_CHECK_ERRORS(cudaMemcpy(img.ptr(), _data->data, elem_size * _size.x, cudaMemcpyDeviceToHost));
    }
    else if (_ndims == 2)
    {
        CUDA_CHECK_ERRORS(cudaMemcpy2D(img.ptr(), img.step()[1], _data->data, _data->pitch, elem_size * _size.x, _size.y, cudaMemcpyDeviceToHost));
    }
    else
    {
        cudaMemcpy3DParms parms = { 0 };
        parms.dstPtr = make_cudaPitchedPtr((void*)img.ptr(), img.step()[1], elem_size * img.size().x, img.size().y);
        parms.srcPtr = make_cudaPitchedPtr(_data->data, _data->pitch, elem_size * _size.x, _size.y);
        parms.extent = make_cudaExtent(elem_size * _size.x, _size.y, _size.z);
        parms.kind = cudaMemcpyDeviceToHost;

        CUDA_CHECK_ERRORS(cudaMemcpy3D(&parms));
    }
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
