#ifndef __CUDA_IMAGE_H__
#define __CUDA_IMAGE_H__

#include "Image/Vec3.h"

class Image;
class CORE_API CudaImage
{
public:
    CudaImage();
    /// Allocates memory on the GPU and uploads data of the specified image.
    CudaImage(const Image& img);
    /// Allocates memory on the GPU for a image with the specified properties.
    CudaImage(int ndims, const Vec3i& size, int type);
    ~CudaImage();

    /// @remark This does not copy the data, use clone if you want a separate copy.
    CudaImage(const CudaImage& other);
    CudaImage& operator=(const CudaImage& other);

    void create(int ndims, const Vec3i& size, int type);
    void release();

    /// Uploads the data from host to device memory.
    void upload(const Image& img);

    /// Downloads the data from device to host memory.
    void download(Image& img);

private:
    struct CORE_API CudaData
    {
        CudaData(int ndims, const Vec3i& size, size_t elem_size);
        ~CudaData();

        uint8_t* data; 
        size_t pitch;
    };
    typedef std::shared_ptr<CudaData> DataPtr;

    DataPtr _data;
    int _ndims;
    Vec3i _size;

    int _pixel_type;
};

#endif // __CUDA_IMAGE_H__
