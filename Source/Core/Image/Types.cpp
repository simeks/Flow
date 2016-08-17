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

int image::string_to_pixel_type(const char* str)
{
    int pixel_t = image::PixelType_Unknown;
    if (strcmp(str, "uint8") == 0)
    {
        pixel_t = image::PixelType_UInt8;
    }
    else if (strcmp(str, "uint16") == 0)
    {
        pixel_t = image::PixelType_UInt16;
    }
    else if (strcmp(str, "uint32") == 0)
    {
        pixel_t = image::PixelType_UInt32;
    }
    else if (strcmp(str, "float32") == 0)
    {
        pixel_t = image::PixelType_Float32;
    }
    else if (strcmp(str, "float64") == 0)
    {
        pixel_t = image::PixelType_Float64;
    }
    else if (strcmp(str, "vec3f") == 0)
    {
        pixel_t = image::PixelType_Vec3f;
    }
    else if (strcmp(str, "vec3d") == 0)
    {
        pixel_t = image::PixelType_Vec3d;
    }
    else if (strcmp(str, "vec4u8") == 0)
    {
        pixel_t = image::PixelType_Vec4u8;
    }
    else if (strcmp(str, "vec4f") == 0)
    {
        pixel_t = image::PixelType_Vec4f;
    }
    else if (strcmp(str, "vec4d") == 0)
    {
        pixel_t = image::PixelType_Vec4d;
    }

    return pixel_t;
}