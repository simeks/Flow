#ifndef __CORE_RVF_H__
#define __CORE_RVF_H__

#include <Core/Image/Image.h>

namespace rvf
{
    CORE_API Image load_rvf(const std::string& file);
    CORE_API bool save_rvf(const std::string& file, const ImageVec3d& img);
}

#endif // __CORE_RVF_H__
