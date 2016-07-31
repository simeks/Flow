#ifndef __CORE_IMAGE_ITK_H__
#define __CORE_IMAGE_ITK_H__

#include <Core/API.h>
#include "Image.h"

namespace image
{
    CORE_API Image load_image(const std::string& file);
    CORE_API bool save_image(const std::string& file, const Image& image);
}


#endif // __CORE_IMAGE_ITK_H__
