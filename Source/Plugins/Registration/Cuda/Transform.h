#ifndef __REGISTRATION_CUDA_TRANSFORM_H__
#define __REGISTRATION_CUDA_TRANSFORM_H__

#include <Core/Image/Image.h>

namespace cuda
{
    Image transform_image(const Image& source, const ImageVec3d& deformation);
}

#endif // __REGISTRATION_CUDA_TRANSFORM_H__
