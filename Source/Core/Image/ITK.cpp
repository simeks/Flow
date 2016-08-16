#include "Common.h"
#include "Platform/FilePath.h"

#include "Flow/FlowImage.h"
#include "Image/Convert.h"
#include "ITK.h"

#include <fstream>
#include <SimpleITK.h>

namespace sitk = itk::simple;

namespace
{
    template<typename T1, typename T2>
    void vector_to_vec3i(Vec3<T1>& v_out, const std::vector<T2>& v_in)
    {
        assert(v_in.size() <= 3);
        for (int i = 0; i < v_in.size(); ++i)
            v_out[i] = v_in[i];
    }
}

Image image::load_image(const std::string& file)
{
    Image ret;

    try
    {
        sitk::Image img = sitk::ReadImage(file);
        if (img.GetDimension() > 3)
            return Image();

        switch (img.GetPixelID())
        {
        case sitk::sitkFloat64:
        {
            ret.create(img.GetSize(), image::PixelType_Float64, (uint8_t*)img.GetBufferAsDouble());
            break;
        }
        case sitk::sitkFloat32:
        {
            ret.create(img.GetSize(), image::PixelType_Float32, (uint8_t*)img.GetBufferAsFloat());
            break;
        }
        case sitk::sitkUInt8:
        {
            ret.create(img.GetSize(), image::PixelType_UInt8, (uint8_t*)img.GetBufferAsUInt8());
            break;
        }
        case sitk::sitkUInt16:
        {
            ret.create(img.GetSize(), image::PixelType_UInt16, (uint8_t*)img.GetBufferAsUInt16());
            break;
        }
        case sitk::sitkUInt32:
        {
            ret.create(img.GetSize(), image::PixelType_UInt32, (uint8_t*)img.GetBufferAsUInt32());
            break;
        }
        case sitk::sitkVectorFloat64:
        {
            if (img.GetNumberOfComponentsPerPixel() == 3)
            {
                ret.create(img.GetSize(), image::PixelType_Vec3d, (uint8_t*)img.GetBufferAsDouble());
            }
            else if (img.GetNumberOfComponentsPerPixel() == 4)
            {
                ret.create(img.GetSize(), image::PixelType_Vec4d, (uint8_t*)img.GetBufferAsDouble());
            }
            break;
        }
        case sitk::sitkVectorFloat32:
        {
            if (img.GetNumberOfComponentsPerPixel() == 3)
            {
                ret.create(img.GetSize(), image::PixelType_Vec3f, (uint8_t*)img.GetBufferAsFloat());
            }
            else if (img.GetNumberOfComponentsPerPixel() == 4)
            {
                ret.create(img.GetSize(), image::PixelType_Vec4f, (uint8_t*)img.GetBufferAsFloat());
            }
            break;
        }
        case sitk::sitkVectorUInt8:
        {
            if (img.GetNumberOfComponentsPerPixel() == 4)
            {
                ret.create(img.GetSize(), image::PixelType_Vec4u8, (uint8_t*)img.GetBufferAsUInt8());
            }
            break;
        }
        }

        if (!ret.valid())
        {
            console::error("Unsupported image format : PixelId = %d, Component count = %d\n", img.GetPixelID(), img.GetNumberOfComponentsPerPixel());
            return Image();
        }

        Vec3d img_origin(0, 0, 0);
        vector_to_vec3i(img_origin, img.GetOrigin());
        ret.set_origin(img_origin);

        Vec3d img_spacing(1, 1, 1);
        vector_to_vec3i(img_spacing, img.GetSpacing());
        ret.set_spacing(img_spacing);

    }
    catch (sitk::GenericException& e)
    {
        console::error("%s", e.GetDescription());
        return Image();
    }

    return ret;
}

bool image::save_image(const std::string& file, const Image& image)
{
    try
    {
        FilePath path(file);
        bool is_png = path.extension() == "png";

        sitk::Image img;
        uint32_t ndims = image.ndims();

        int pixel_type = image.pixel_type();
        size_t size_in_bytes = image::pixel_size(pixel_type);

        std::vector<uint32_t> size;
        for (uint32_t i = 0; i < ndims; ++i)
        {
            size.push_back(image.size()[i]);
            size_in_bytes *= image.size()[i];
        }

        if (is_png)
        {
            // TODO: Support for higher bit depth than 8bits per channel?

            Image src = image;
            uint8_t* dest = nullptr;
            if (pixel_type == image::PixelType_UInt8)
            {
                img = sitk::Image(size, sitk::sitkUInt8);
                dest = img.GetBufferAsUInt8();
            }
            else if (pixel_type == image::PixelType_UInt16)
            {
                img = sitk::Image(size, sitk::sitkUInt16);
                dest = (uint8_t*)img.GetBufferAsUInt16();
            }
            else if (pixel_type == image::PixelType_Float32 || 
                pixel_type == image::PixelType_Float64)
            {
                src = image::convert_image(image, image::PixelType_UInt8, 255.0, 0.0);

                img = sitk::Image(size, sitk::sitkUInt8);
                dest = img.GetBufferAsUInt8();
            }
            else if (pixel_type == image::PixelType_Vec4f || 
                     pixel_type == image::PixelType_Vec4d)
            {
                src = image::convert_image(image, image::PixelType_Vec4u8, 255.0, 0.0);

                img = sitk::Image(size, sitk::sitkVectorUInt8, 4);
                dest = (uint8_t*)img.GetBufferAsUInt8();
            }
            else
            {
                console::error("Could not save as PNG, unsupported format (PixelType: %d).\n", pixel_type);
                return false;
            }

            src.copy_to(dest);
        }
        else
        {
            uint8_t* dest = nullptr;
            if (pixel_type == image::PixelType_UInt8)
            {
                img = sitk::Image(size, sitk::sitkUInt8);
                dest = img.GetBufferAsUInt8();
            }
            else if (pixel_type == image::PixelType_UInt16)
            {
                img = sitk::Image(size, sitk::sitkUInt16);
                dest = (uint8_t*)img.GetBufferAsUInt16();
            }
            else if (pixel_type == image::PixelType_UInt32)
            {
                img = sitk::Image(size, sitk::sitkUInt32);
                dest = (uint8_t*)img.GetBufferAsUInt32();
            }
            else if (pixel_type == image::PixelType_Float32)
            {
                img = sitk::Image(size, sitk::sitkFloat32);
                dest = (uint8_t*)img.GetBufferAsFloat();
            }
            else if (pixel_type == image::PixelType_Float64)
            {
                img = sitk::Image(size, sitk::sitkFloat64);
                dest = (uint8_t*)img.GetBufferAsDouble();
            }
            else if (pixel_type == image::PixelType_Vec3f)
            {
                img = sitk::Image(size, sitk::sitkVectorFloat32, 3);
                dest = (uint8_t*)img.GetBufferAsFloat();
            }
            else if (pixel_type == image::PixelType_Vec3d)
            {
                img = sitk::Image(size, sitk::sitkVectorFloat64, 3);
                dest = (uint8_t*)img.GetBufferAsDouble();
            }
            else if (pixel_type == image::PixelType_Vec4u8)
            {
                img = sitk::Image(size, sitk::sitkVectorUInt8, 4);
                dest = (uint8_t*)img.GetBufferAsUInt8();
            }
            else if (pixel_type == image::PixelType_Vec4f)
            {
                img = sitk::Image(size, sitk::sitkVectorFloat32, 4);
                dest = (uint8_t*)img.GetBufferAsFloat();
            }
            else if (pixel_type == image::PixelType_Vec4d)
            {
                img = sitk::Image(size, sitk::sitkVectorFloat64, 4);
                dest = (uint8_t*)img.GetBufferAsDouble();
            }
            else
            {
                return false;
            }

            image.copy_to(dest);
        }

        std::vector<double> origin;
        std::vector<double> spacing;
        for (uint32_t i = 0; i < ndims; ++i)
        {
            origin.push_back(image.origin()[i]);
            spacing.push_back(image.spacing()[i]);
        }
        img.SetOrigin(origin);
        img.SetSpacing(spacing);

        sitk::WriteImage(img, file);
    }
    catch (sitk::GenericException& e)
    {
        console::error("%s", e.GetDescription());
        return false;
    }
    return true;
}