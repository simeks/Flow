#include <Core/Common.h>
#include <Core/Cuda/Cuda.h>
#include <Core/Cuda/CudaImage.h>
#include <Core/Image/Image.h>

#include <Tools/Testing/Framework.h>

using namespace testing;

TEST_CASE(CudaImage_transfer)
{
    CudaImage img_1d;
    img_1d.create(1, Vec3i(1000, 1, 1), image::PixelType_UInt8);

    CudaImage img_2d;
    img_2d.create(2, Vec3i(1000, 1000, 1), image::PixelType_UInt8);

    CudaImage img_3d;
    img_3d.create(3, Vec3i(1000, 1000, 1000), image::PixelType_UInt8);

    ASSERT_EXPR(true);
}

template<typename T> bool test_upload_download(int ndims, const Vec3i& size)
{
    ImageTpl<T> img(ndims, size);

    for (int z = 0; z < size.z; ++z)
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x)
                img(x, y, z) = rand() % std::numeric_limits<T>::max();

    CudaImage cuda_img;
    cuda_img.upload(img);

    ImageTpl<T> img_2;
    cuda_img.download(img_2);

    for (int z = 0; z < size.z; ++z)
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x)
                if (img_2(x, y, z) != img(x, y, z))
                    return false;
    return true;
}

TEST_CASE(CudaImage_upload_download)
{
    ASSERT_EXPR(test_upload_download<uint8_t>(1, Vec3i(500, 1, 1)));
    ASSERT_EXPR(test_upload_download<uint8_t>(2, Vec3i(500, 500, 1)));
    ASSERT_EXPR(test_upload_download<uint8_t>(3, Vec3i(500, 500, 500)));

    ASSERT_EXPR(test_upload_download<uint16_t>(1, Vec3i(500, 1, 1)));
    ASSERT_EXPR(test_upload_download<uint16_t>(2, Vec3i(500, 500, 1)));
    ASSERT_EXPR(test_upload_download<uint16_t>(3, Vec3i(500, 500, 500)));

    ASSERT_EXPR(test_upload_download<uint32_t>(1, Vec3i(500, 1, 1)));
    ASSERT_EXPR(test_upload_download<uint32_t>(2, Vec3i(500, 500, 1)));
    ASSERT_EXPR(test_upload_download<uint32_t>(3, Vec3i(500, 500, 500)));
}
