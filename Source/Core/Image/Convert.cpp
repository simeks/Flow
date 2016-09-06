#include "Common.h"

#include "Convert.h"

#include <limits.h>
#include <float.h>

// TODO: Make global
template<typename T> INLINE T number_cast(uint8_t v) { return T(v); }
template<typename T> INLINE T number_cast(uint16_t v) { return T(v); }
template<typename T> INLINE T number_cast(uint32_t v) { return T(v); }
template<typename T> INLINE T number_cast(float v) { return T(v); }
template<typename T> INLINE T number_cast(double v) { return T(v); }

template<> INLINE uint8_t number_cast<uint8_t>(uint16_t v) { return (uint8_t)std::min((uint32_t)v, (uint32_t)UCHAR_MAX); }
template<> INLINE uint8_t number_cast<uint8_t>(uint32_t v) { return (uint8_t)std::min(v, (uint32_t)UCHAR_MAX); }
template<> INLINE uint8_t number_cast<uint8_t>(float v) { int iv = (int)round(v); return (uint8_t)std::min((uint32_t)std::max(iv, 0), (uint32_t)UCHAR_MAX); }
template<> INLINE uint8_t number_cast<uint8_t>(double v) { int iv = (int)round(v); return (uint8_t)std::min((uint32_t)std::max(iv, 0), (uint32_t)UCHAR_MAX); }

template<> INLINE uint16_t number_cast<uint16_t>(uint32_t v) { return (uint16_t)std::min(v, (uint32_t)USHRT_MAX); }
template<> INLINE uint16_t number_cast<uint16_t>(float v) { int iv = (int)round(v); return (uint16_t)std::min((uint32_t)std::max(iv, 0), (uint32_t)USHRT_MAX); }
template<> INLINE uint16_t number_cast<uint16_t>(double v) { int iv = (int)round(v); return (uint16_t)std::min((uint32_t)std::max(iv, 0), (uint32_t)USHRT_MAX); }

template<> INLINE uint32_t number_cast<uint32_t>(float v) { return (uint32_t)round(v); }
template<> INLINE uint32_t number_cast<uint32_t>(double v) { return (uint32_t)round(v); }

template<typename TSrc, typename TDest>
static INLINE void convert_image_tpl(
    const TSrc* src,
    TDest* dest,
    const size_t* src_step,
    const size_t* dest_step,
    const Vec3i& size,
    size_t n_comp)
{
    size_t sstep[3];
    size_t dstep[3];
    for (int i = 0; i < 3; ++i)
    {
        sstep[i] = src_step[i] / sizeof(*src);
        dstep[i] = dest_step[i] / sizeof(*dest);
    }

    for (int z = 0; z < size.z; ++z)
    {
        for (int y = 0; y < size.y; ++y)
        {
            const TSrc* s = src + (z*sstep[2] + y*sstep[1]);
            TDest* d = dest + (z*dstep[2] + y*dstep[1]);
            for (size_t x = 0; x < size.x*n_comp; ++x)
            {
                d[x] = number_cast<TDest>(s[x]);
            }
        }
    }
}

template<typename TSrc, typename TDest>
static INLINE void convert_image_scale_tpl(
    const TSrc* src,
    TDest* dest,
    const size_t* src_step,
    const size_t* dest_step,
    const Vec3i& size,
    size_t n_comp,
    double scale,
    double shift)
{
    size_t sstep[3];
    size_t dstep[3];
    for (int i = 0; i < 3; ++i)
    {
        sstep[i] = src_step[i] / sizeof(*src);
        dstep[i] = dest_step[i] / sizeof(*dest);
    }

    for (int z = 0; z < size.z; ++z)
    {
        for (int y = 0; y < size.y; ++y)
        {
            const TSrc* s = src + (z*sstep[2] + y*sstep[1]);
            TDest* d = dest + (z*dstep[2] + y*dstep[1]);
            for (size_t x = 0; x < size.x*n_comp; ++x)
            {
                d[x] = number_cast<TDest>(s[x] * scale + shift);
            }
        }
    }
}

Image image::convert_image(const Image& src, int pixel_type, double scale, double shift)
{
    bool do_scale = fabs(scale - 1.0) > DBL_EPSILON || fabs(shift) > DBL_EPSILON;
    if (!do_scale && src.pixel_type() == pixel_type)
        return src.clone();

    Image result;
    if (src.pixel_type() == image::PixelType_Vec4u8 && pixel_type == image::PixelType_Vec4f)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<uint8_t, float>(src.ptr<uint8_t>(), result.ptr<float>(), src.step(), result.step(), result.size(), 4, scale, shift);
        else
            convert_image_tpl<uint8_t, float>(src.ptr<uint8_t>(), result.ptr<float>(), src.step(), result.step(), result.size(), 4);
    }
    else if (src.pixel_type() == image::PixelType_Vec4f && pixel_type == image::PixelType_Vec4u8)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<float, uint8_t>(src.ptr<float>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 4, scale, shift);
        else
            convert_image_tpl<float, uint8_t>(src.ptr<float>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 4);
    }
    else if (src.pixel_type() == image::PixelType_Vec4u8 && pixel_type == image::PixelType_Vec4d)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<uint8_t, double>(src.ptr<uint8_t>(), result.ptr<double>(), src.step(), result.step(), result.size(), 4, scale, shift);
        else
            convert_image_tpl<uint8_t, double>(src.ptr<uint8_t>(), result.ptr<double>(), src.step(), result.step(), result.size(), 4);
    }
    else if (src.pixel_type() == image::PixelType_Vec4d && pixel_type == image::PixelType_Vec4u8)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<double, uint8_t>(src.ptr<double>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 4, scale, shift);
        else
            convert_image_tpl<double, uint8_t>(src.ptr<double>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 4);
    }
    else if (src.pixel_type() == image::PixelType_Float32 && pixel_type == image::PixelType_Float64)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<float, double>(src.ptr<float>(), result.ptr<double>(), src.step(), result.step(), result.size(), 1, scale, shift);
        else
            convert_image_tpl<float, double>(src.ptr<float>(), result.ptr<double>(), src.step(), result.step(), result.size(), 1);
    }
    else if (src.pixel_type() == image::PixelType_Float64 && pixel_type == image::PixelType_Float32)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<double, float>(src.ptr<double>(), result.ptr<float>(), src.step(), result.step(), result.size(), 1, scale, shift);
        else
            convert_image_tpl<double, float>(src.ptr<double>(), result.ptr<float>(), src.step(), result.step(), result.size(), 1);
    }
    else if (src.pixel_type() == image::PixelType_Float32 && pixel_type == image::PixelType_UInt8)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<float, uint8_t>(src.ptr<float>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 1, scale, shift);
        else
            convert_image_tpl<float, uint8_t>(src.ptr<float>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 1);
    }
    else if (src.pixel_type() == image::PixelType_UInt8 && pixel_type == image::PixelType_Float32)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<uint8_t, float>(src.ptr<uint8_t>(), result.ptr<float>(), src.step(), result.step(), result.size(), 1, scale, shift);
        else
            convert_image_tpl<uint8_t, float>(src.ptr<uint8_t>(), result.ptr<float>(), src.step(), result.step(), result.size(), 1);
    }
    else if (src.pixel_type() == image::PixelType_Float64 && pixel_type == image::PixelType_UInt8)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<double, uint8_t>(src.ptr<double>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 1, scale, shift);
        else
            convert_image_tpl<double, uint8_t>(src.ptr<double>(), result.ptr<uint8_t>(), src.step(), result.step(), result.size(), 1);
    }
    else if (src.pixel_type() == image::PixelType_UInt8 && pixel_type == image::PixelType_Float64)
    {
        result = Image(src.ndims(), src.size(), pixel_type);
        if (do_scale)
            convert_image_scale_tpl<uint8_t, double>(src.ptr<uint8_t>(), result.ptr<double>(), src.step(), result.step(), result.size(), 1, scale, shift);
        else
            convert_image_tpl<uint8_t, double>(src.ptr<uint8_t>(), result.ptr<double>(), src.step(), result.step(), result.size(), 1);
    }
    else
    {
        assert(false);
        FATAL_ERROR("Conversion of specified type not supported.");
    }

    result.set_spacing(src.spacing());
    result.set_origin(src.origin());

    return result;
}
