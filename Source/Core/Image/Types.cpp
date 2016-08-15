#include "Common.h"

#include "Types.h"
#include "Vec3.h"

size_t image::pixel_size(int type)
{
    switch (type)
    {
    case PixelType_UInt8:
        return sizeof(uint8_t);
    case PixelType_UInt16:
        return sizeof(uint16_t);
    case PixelType_UInt32:
        return sizeof(uint32_t);
    case PixelType_Float32:
        return sizeof(float);
    case PixelType_Float64:
        return sizeof(double);
    case PixelType_Vec3f:
        return sizeof(Vec3f);
    case PixelType_Vec3d:
        return sizeof(Vec3d);
    case PixelType_Vec4u8:
        return sizeof(uint8_t)*4;
    case PixelType_Vec4f:
        return sizeof(float) * 4;
    case PixelType_Vec4d:
        return sizeof(double) * 4;
    }
    return 0;
}
