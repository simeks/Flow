#ifndef __CORE_IMAGE_CONVERT_H__
#define __CORE_IMAGE_CONVERT_H__

#include "Image.h"

namespace image
{
    Image convert_image(const Image& src, int pixel_type, double scale = 1.0, double shift = 0.0);
}

#endif // __CORE_IMAGE_CONVERT_H__
