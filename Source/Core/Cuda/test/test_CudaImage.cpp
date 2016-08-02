#include <Core/Common.h>
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