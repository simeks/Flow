#ifndef __CORE_IMAGE_NORMALIZE_H__
#define __CORE_IMAGE_NORMALIZE_H__

#include "Image.h"

namespace image
{
    Image normalize_image(const Image& src, double min = 0.0, double max = 1.0);
}

#endif // __CORE_IMAGE_NORMALIZE_H__
