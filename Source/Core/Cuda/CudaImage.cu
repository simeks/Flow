#include "Common.h"
#include "Cuda.h"
#include "CudaImage.h"

#include "Image/Image.h"

namespace
{
    cudaChannelFormatDesc create_channel_desc(int pixel_type)
    {
        switch (pixel_type)
        {
        case image::PixelType_UInt8:
        case image::PixelType_UInt16:
        case image::PixelType_UInt32:
        {
            int e = (int)image::pixel_size(pixel_type) * 8;
            return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
        }
        case image::PixelType_Float32:
        case image::PixelType_Float64:
        {
            int e = (int)image::pixel_size(pixel_type) * 8;
            return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
        }
        case image::PixelType_Vec3u8:
        {
            int e = (int)sizeof(uint8_t) * 8;
            return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindUnsigned);
        }
        case image::PixelType_Vec3f:
        {
            int e = (int)sizeof(float) * 8;
            return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindFloat);
        }
        case image::PixelType_Vec3d:
        {
            int e = (int)sizeof(double) * 8;
            return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindFloat);
        }
        case image::PixelType_Vec4u8:
        {
            int e = (int)sizeof(uint8_t) * 8;
            return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
        }
        case image::PixelType_Vec4f:
        {
            int e = (int)sizeof(float) * 8;
            return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
        }
        case image::PixelType_Vec4d:
        {
            int e = (int)sizeof(double) * 8;
            return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
        }
        }
        return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
    }
}

CudaImage::CudaData::CudaData(int ndims, const Vec3i& size, int type, int flags) :
    data(nullptr),
    arr(nullptr)
{
    if (flags & Flag_Texture)
    {
        // There are certain memory requirements (padding, etc) for images that are to be bound as textures

        cudaChannelFormatDesc desc = create_channel_desc(type);
        if (ndims == 3)
        {
            CUDA_CHECK_ERRORS(cudaMalloc3DArray(&arr, &desc, make_cudaExtent(size.x, size.y, size.z)));
        }
        else
        {
            CUDA_CHECK_ERRORS(cudaMallocArray(&arr, &desc, size.x, size.y));
        }
    }
    else
    {
        // Otherwise we just allocate it linearly
        size_t elem_size = image::pixel_size(type);
        CUDA_CHECK_ERRORS(cudaMalloc(&data, elem_size * size.x * size.y * size.z));
    }

}
CudaImage::CudaData::~CudaData()
{
    if (data)
        cudaFree(data);
    if (arr)
        cudaFreeArray(arr);
    data = nullptr;
    arr = nullptr;
}

CudaImage::CudaImage() :
_data(nullptr),
_ndims(0),
_pixel_type(image::PixelType_Unknown),
_flags(0)
{
}
CudaImage::CudaImage(const Image& img, int flags)
{
    upload(img, flags);
}
CudaImage::CudaImage(int ndims, const Vec3i& size, int type, int flags)
{
    create(ndims, size, type, flags);
}
CudaImage::~CudaImage()
{
    _data.reset();
}

void CudaImage::create(int ndims, const Vec3i& size, int type, int flags)
{
    assert(ndims <= 3);
    assert(type != image::PixelType_Unknown);

    if (_ndims == ndims && 
        _size == size && 
        _pixel_type == type && 
        _flags == flags && 
        _data)
        return;

    release();

    _ndims = ndims;
    _size = size;
    _pixel_type = type;
    _flags = flags;

    _data = std::make_shared<CudaData>(ndims, size, type, flags);
}
void CudaImage::release()
{
    _pixel_type = image::PixelType_Unknown;
    _data.reset();
}
void CudaImage::upload(const Image& img, int flags)
{
    create(img.ndims(), img.size(), img.pixel_type(), flags);

    size_t elem_size = image::pixel_size(_pixel_type);
    if (flags & Flag_Texture)
    {
        if (_ndims == 3)
        {
            cudaMemcpy3DParms params = { 0 };
            params.srcPtr = make_cudaPitchedPtr((void*)img.ptr(), img.size().x*elem_size, img.size().x, img.size().y);
            params.dstArray = _data->arr;
            params.extent = make_cudaExtent(img.size().x, img.size().y, img.size().z);
            params.kind = cudaMemcpyHostToDevice;
            CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));
        }
        else
        {
            CUDA_CHECK_ERRORS(cudaMemcpyToArray(_data->arr, 0, 0, img.ptr(),
                elem_size * _size.x * _size.y * _size.z, cudaMemcpyHostToDevice));
        }
    }
    else
    {
        CUDA_CHECK_ERRORS(cudaMemcpy(_data->data, img.ptr(), 
            elem_size * _size.x * _size.y * _size.z, cudaMemcpyHostToDevice));
    }
}
void CudaImage::download(Image& img)
{
    img.create(_ndims, _size, _pixel_type);

    size_t elem_size = image::pixel_size(_pixel_type);
    if (_flags & Flag_Texture)
    {
        if (_ndims == 3)
        {
            cudaMemcpy3DParms params = { 0 };
            params.dstPtr = make_cudaPitchedPtr((void*)img.ptr(), img.size().x*elem_size, img.size().x, img.size().y);
            params.srcArray = _data->arr;
            params.extent = make_cudaExtent(img.size().x, img.size().y, img.size().z);
            params.kind = cudaMemcpyDeviceToHost;
            CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));
        }
        else
        {
            CUDA_CHECK_ERRORS(cudaMemcpyFromArray(img.ptr(), _data->arr, 0, 0,
                elem_size * _size.x * _size.y * _size.z, cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        CUDA_CHECK_ERRORS(cudaMemcpy(img.ptr(), _data->data, 
            elem_size * _size.x * _size.y * _size.z, cudaMemcpyDeviceToHost));
    }
}
int CudaImage::flags() const
{
    return _flags;
}
uint8_t* CudaImage::ptr()
{
    assert(_data);
    return _data->data;
}
cudaArray_t CudaImage::cuda_array()
{
    assert(_data);
    return _data->arr;
}
cudaExtent CudaImage::cuda_extent()
{
    return make_cudaExtent(_size.x, _size.y, _size.z);
}
CudaImage::CudaImage(const CudaImage& other) :
_data(other._data),
_ndims(other._ndims),
_size(other._size),
_pixel_type(other._pixel_type),
_flags(other._flags)
{
}
CudaImage& CudaImage::operator=(const CudaImage& other)
{
    _data = other._data;
    _ndims = other._ndims;
    _size = other._size;
    _pixel_type = other._pixel_type;
    _flags = other._flags;

    return *this;
}
