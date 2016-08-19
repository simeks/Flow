#include "Common.h"

#include "Normalize.h"
#include "Image.h"

template<typename TImage>
static TImage normalize_image(const TImage& src, double min, double max)
{
    Vec3i dims = src.size();

    double in_min, in_max;
    image::find_min_max(src, in_min, in_max);

    TImage res(src.ndims(), src.size());

#pragma omp parallel for
    for (int z = 0; z < dims.z; ++z)
    {
        for (int y = 0; y < dims.y; ++y)
        {
            for (int x = 0; x < dims.x; ++x)
            {
                res(x, y, z) = TImage::TPixelType((max - min) * (src(x, y, z) - in_min) / (in_max - in_min) + min);
            }
        }
    }

    return res;
}

Image image::normalize_image(const Image& src, double min, double max)
{
    switch (src.pixel_type())
    {
    case image::PixelType_UInt8:
        return ::normalize_image<ImageUInt8>(src, min, max);
    case image::PixelType_UInt16:
        return ::normalize_image<ImageUInt16>(src, min, max);
    case image::PixelType_UInt32:
        return ::normalize_image<ImageUInt32>(src, min, max);
    case image::PixelType_Float32:
        return ::normalize_image<ImageFloat32>(src, min, max);
    case image::PixelType_Float64:
        return ::normalize_image<ImageFloat64>(src, min, max);
    case image::PixelType_Vec3u8:
        return ::normalize_image<ImageVec3u8>(src, min, max);
    case image::PixelType_Vec3f:
        return ::normalize_image<ImageVec3f>(src, min, max);
    case image::PixelType_Vec3d:
        return ::normalize_image<ImageVec3d>(src, min, max);
    case image::PixelType_Vec4f:
        return ::normalize_image<ImageColorf>(src, min, max);
    default:
        assert(false);
    }
    return Image();
}

