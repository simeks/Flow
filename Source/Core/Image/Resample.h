#ifndef __CORE_IMAGE_RESAMPLE_H__
#define __CORE_IMAGE_RESAMPLE_H__

namespace image
{
    CORE_API Image downsample_image(const Image& img, double scale);
    CORE_API Image downsample_image_gaussian(const Image& img, double scale, double sigma);

    CORE_API ImageVec3d downsample_vectorfield(const ImageVec3d& field, double scale, ImageVec3d& residual);

    CORE_API ImageVec3d upsample_vectorfield(const ImageVec3d& field, double scale);
    CORE_API void upsample_vectorfield(const ImageVec3d& field, double scale, const ImageVec3d& residual, ImageVec3d& out);
}

#endif // __CORE_IMAGE_RESAMPLE_H__
