#include "Common.h"

#include "Image.h"
#include "GaussianFilter.h"

template<typename TImage>
static TImage gaussian_filter_3d(const TImage& img, double sigma)
{
    assert(img.ndims() == 3);

    TImage result = img.clone();

    double sx = img.spacing().x;
    double sy = img.spacing().y;
    double sz = img.spacing().z;

    double factor = (-1.0) / (2 * sigma*sigma);

    Vec3i dims = img.size();

    // X dimension
    int size = (int)ceil(3 * sigma / sx);
#pragma omp parallel for
    for (int y = 0; y < dims.y; y++)
    {
        for (int z = 0; z < dims.z; z++)
        {
            for (int x = 0; x < dims.x; x++)
            {
                typename TImage::TPixelType value(0);
                double norm = 0;
                for (int t = -size; t < size + 1; t++)
                {
                    double c = exp(factor*(t*sx)*(t*sx));

                    int sx = std::min(std::max(x + t, 0), dims.x - 1);
                    value += typename TImage::TPixelType(c*result(sx, y, z));
                    norm += c;
                }
                result(x, y, z) = typename TImage::TPixelType(value / norm);
            }
        }
    }

    //Y dimension
    size = (int)ceil(3 * sigma / sy);
#pragma omp parallel for
    for (int x = 0; x < dims.x; x++)
    {
        for (int z = 0; z < dims.z; z++)
        {
            for (int y = 0; y < dims.y; y++)
            {
                typename TImage::TPixelType value(0);
                double norm = 0;
                for (int t = -size; t < size + 1; t++)
                {
                    double c = exp(factor*(t*sx)*(t*sx));

                    int sy = std::min(std::max(y + t, 0), dims.y - 1);
                    value += typename TImage::TPixelType(c*result(x, sy, z));
                    norm += c;
                }
                result(x, y, z) = typename TImage::TPixelType(value / norm);
            }
        }
    }

    //Z dimension
    size = (int)ceil(3 * sigma / sz);
#pragma omp parallel for
    for (int x = 0; x < dims.x; x++)
    {
        for (int y = 0; y < dims.y; y++)
        {
            for (int z = 0; z < dims.z; z++)
            {
                typename TImage::TPixelType value(0);
                double norm = 0;
                for (int t = -size; t < size + 1; t++)
                {
                    double c = exp(factor*(t*sx)*(t*sx));

                    int sz = std::min(std::max(z + t, 0), dims.z - 1);
                    value += typename TImage::TPixelType(c*result(x, y, sz));
                    norm += c;
                }
                result(x, y, z) = typename TImage::TPixelType(value / norm);
            }
        }
    }
    return result;
}

template<typename TImage>
static TImage gaussian_filter_2d(const TImage& img, double sigma)
{
    TImage result = img.clone();

    double sx = img.spacing().x;
    double sy = img.spacing().y;

    double factor = (-1.0) / (2 * sigma*sigma);

    Vec3i dims = img.size();

    // X dimension
    int size = (int)ceil(3 * sigma / sx);
#pragma omp parallel for
    for (int y = 0; y < dims.y; y++)
    {
        for (int x = 0; x < dims.x; x++)
        {
            typename TImage::TPixelType value(0);
            double norm = 0;
            for (int t = -size; t < size + 1; t++)
            {
                double c = exp(factor*(t*sx)*(t*sx));

                int sx = std::min(std::max(x + t, 0), dims.x - 1);
                value += typename TImage::TPixelType(c*result(sx, y));
                norm += c;
            }
            result(x, y) = typename TImage::TPixelType(value / norm);
        }
    }

    //Y dimension
    size = (int)ceil(3 * sigma / sy);
#pragma omp parallel for
    for (int x = 0; x < dims.x; x++)
    {
        for (int y = 0; y < dims.y; y++)
        {
            typename TImage::TPixelType value(0);
            double norm = 0;
            for (int t = -size; t < size + 1; t++)
            {
                double c = exp(factor*(t*sx)*(t*sx));

                int sy = std::min(std::max(y + t, 0), dims.y - 1);
                value += typename TImage::TPixelType(c*result(x, sy));
                norm += c;
            }
            result(x, y) = typename TImage::TPixelType(value / norm);
        }
    }

    return result;
}


Image image::gaussian_filter(const Image& img, double sigma)
{
    if (img.ndims() == 3)
    {
        switch (img.pixel_type())
        {
        case image::PixelType_UInt32:
            return gaussian_filter_3d<ImageUInt32>(img, sigma);
        case image::PixelType_Float32:
            return gaussian_filter_3d<ImageFloat32>(img, sigma);
        case image::PixelType_Float64:
            return gaussian_filter_3d<ImageFloat64>(img, sigma);
        case image::PixelType_Vec3f:
            return gaussian_filter_3d<ImageVec3f>(img, sigma);
        case image::PixelType_Vec3d:
            return gaussian_filter_3d<ImageVec3d>(img, sigma);
        case image::PixelType_Vec4f:
            return gaussian_filter_3d<ImageColorf>(img, sigma);
        default:
            assert(false);
        }
    }
    else
    {
        switch (img.pixel_type())
        {
        case image::PixelType_UInt32:
            return gaussian_filter_2d<ImageUInt32>(img, sigma);
        case image::PixelType_Float32:
            return gaussian_filter_2d<ImageFloat32>(img, sigma);
        case image::PixelType_Float64:
            return gaussian_filter_2d<ImageFloat64>(img, sigma);
        case image::PixelType_Vec3f:
            return gaussian_filter_2d<ImageVec3f>(img, sigma);
        case image::PixelType_Vec3d:
            return gaussian_filter_2d<ImageVec3d>(img, sigma);
        case image::PixelType_Vec4f:
            return gaussian_filter_2d<ImageColorf>(img, sigma);
        default:
            assert(false);
        }
    }
    return Image();
}
