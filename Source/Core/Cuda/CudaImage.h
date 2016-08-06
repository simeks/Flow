#ifndef __CUDA_IMAGE_H__
#define __CUDA_IMAGE_H__

#include <Core/Image/Vec3.h>

class Image;
class CORE_API CudaImage
{
public:
    enum Flags
    {
        Flag_Texture = 1 //< Indicates that this object is to be used as a read-only texture.
    };

    CudaImage();
    /// Allocates memory on the GPU and uploads data of the specified image.
    CudaImage(const Image& img, int flags = 0);
    /// Allocates memory on the GPU for a image with the specified properties.
    CudaImage(int ndims, const Vec3i& size, int type, int flags = 0);
    ~CudaImage();

    /// @remark This does not copy the data, use clone if you want a separate copy.
    CudaImage(const CudaImage& other);
    CudaImage& operator=(const CudaImage& other);

    void create(int ndims, const Vec3i& size, int type, int flags = 0);
    void release();

    /// Uploads the data from host to device memory.
    void upload(const Image& img, int flags = 0);

    /// Downloads the data from device to host memory.
    void download(Image& img);

    int flags() const;

    uint8_t* ptr();
    cudaArray_t cuda_array();
    cudaExtent cuda_extent();

    template<typename T> 
    T* ptr() { return (T*)ptr(); }
    
private:
    struct CORE_API CudaData
    {
        CudaData(int ndims, const Vec3i& size, int type, int flags = 0);
        ~CudaData();

        uint8_t* data;
        cudaArray_t arr;
        
    };
    typedef std::shared_ptr<CudaData> DataPtr;

    DataPtr _data;
    int _ndims;
    Vec3i _size;

    int _pixel_type;
    int _flags;
};

#endif // __CUDA_IMAGE_H__
