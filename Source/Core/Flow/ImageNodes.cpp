#include "Common.h"

#include "FlowContext.h"
#include "FlowImage.h"
#include "FlowNode.h"
#include "FlowPrimitives.h"
#include "FlowString.h"
#include "FlowSystem.h"
#include "FlowVector.h"
#include "Image/Convert.h"
#include "Image/Image.h"
#include "Image/ITK.h"
#include "Image/Normalize.h"
#include "ImageNodes.h"
#include "Platform/FilePath.h"
#include "RVF/RVF.h"

class ImageLoadNode : public FlowNode
{
    DECLARE_OBJECT(ImageLoadNode, FlowNode);
public:
    ImageLoadNode()
    {
        add_pin("Filename", FlowPin::In);
        add_pin("Image", FlowPin::Out);
    }

    void run(FlowContext& context) OVERRIDE
    {
        FilePath path(context.read_string("Filename"));
            
        FlowImage* result = nullptr;
        if (path.extension() == "rvf")
        {
            Image img = rvf::load_rvf(path.get());
            if (img)
            {
                result = new FlowImage(img);
            }
        }
        else
        {
            Image img = image::load_image(path.get());
            if (img)
            {
                result = new FlowImage(img);
            }
        }

        if(result)
            context.write_pin("Image", result);
    }
    const char* title() const OVERRIDE
    {
        return "LoadImage";
    }
    const char* category() const OVERRIDE
    {
        return "Image";
    }

};
IMPLEMENT_OBJECT(ImageLoadNode, "ImageLoadNode");

class ImageSaveNode : public FlowNode
{
    DECLARE_OBJECT(ImageSaveNode, FlowNode);
public:
    ImageSaveNode()
    {
        add_pin("Image", FlowPin::In);
        add_pin("Filename", FlowPin::In);
    }

    void run(FlowContext& context) OVERRIDE
    {
        FlowImage* img = context.read_pin<FlowImage>("Image");
        FilePath path = context.read_string("Filename");

        if (img)
        {
            if (path.extension() == "rvf")
            {
                if (img->pixel_type() == image::PixelType_Vec3d)
                    rvf::save_rvf(path.get(), img->image());
                else
                    FATAL_ERROR("Invalid image format for RVF files.");
            }
            else
            {
                image::save_image(path.get(), img->image());
            }
        }
    }
    const char* title() const OVERRIDE
    {
        return "SaveImage";
    }
    const char* category() const OVERRIDE
    {
        return "Image";
    }

};
IMPLEMENT_OBJECT(ImageSaveNode, "ImageSaveNode");


class ImagePropertiesNode : public FlowNode
{
    DECLARE_OBJECT(ImagePropertiesNode, FlowNode);
public:
    ImagePropertiesNode()
    {
        add_pin("Image", FlowPin::In);
        add_pin("Size", FlowPin::Out);
        add_pin("Spacing", FlowPin::Out);
        add_pin("Origin", FlowPin::Out);
    }

    void run(FlowContext& context) OVERRIDE
    {
        FlowImage* img = context.read_pin<FlowImage>("Image");
        if (img)
        {
            if (pin("Size")->is_linked())
            {
                context.write_pin("Size", new FlowVec3i(img->size()));
            }
            if (pin("Spacing")->is_linked())
            {
                context.write_pin("Spacing", new FlowVec3d(img->spacing()));
            }
            if (pin("Origin")->is_linked())
            {
                context.write_pin("Origin", new FlowVec3d(img->origin()));
            }
        }
    }
    const char* title() const OVERRIDE
    {
        return "Properties";
    }
    const char* category() const OVERRIDE
    {
        return "Image";
    }

};
IMPLEMENT_OBJECT(ImagePropertiesNode, "ImagePropertiesNode");

class ImageConvertNode : public FlowNode
{
    DECLARE_OBJECT(ImageConvertNode, FlowNode);
public:
    ImageConvertNode()
    {
        add_pin("In", FlowPin::In);
        add_pin("Format", FlowPin::In);
        add_pin("Scale", FlowPin::In);
        add_pin("Shift", FlowPin::In);
        add_pin("Out", FlowPin::Out);
    }

    void run(FlowContext& context) OVERRIDE
    {
        FlowImage* img = context.read_pin<FlowImage>("In");
        FlowObject* fmt = context.read_pin<FlowObject>("Format");
        if (img && fmt)
        {
            int target_type = image::PixelType_Unknown;
            if (fmt->is_a(NumericObject::static_class()))
            {
                target_type = (int)((NumericObject*)fmt)->as_int();
            }
            else if (fmt->is_a(FlowString::static_class()))
            {
                target_type = image::string_to_pixel_type(((FlowString*)fmt)->get().c_str());
            }

            if (target_type == image::PixelType_Unknown)
                FATAL_ERROR("Failed to convert image. No supported target format was specified\n");

            double scale = 1.0;
            double shift = 0.0;

            NumericObject* n = context.read_pin<NumericObject>("Scale");
            if (n)
                scale = n->as_float();

            n = context.read_pin<NumericObject>("Shift");
            if (n)
                shift = n->as_float();

            Image converted = image::convert_image(img->image(), target_type, scale, shift);
            context.write_pin("Out", new FlowImage(converted));
        }
    }
        
    const char* title() const OVERRIDE
    {
        return "Convert";
    }
    const char* category() const OVERRIDE
    {
        return "Image";
    }

};
IMPLEMENT_OBJECT(ImageConvertNode, "ImageConvertNode");


class ImageNormalizeNode : public FlowNode
{
    DECLARE_OBJECT(ImageNormalizeNode, FlowNode);
public:
    ImageNormalizeNode()
    {
        add_pin("In", FlowPin::In);
        add_pin("Min", FlowPin::In);
        add_pin("Max", FlowPin::In);
        add_pin("Out", FlowPin::Out);
    }

    void run(FlowContext& context) OVERRIDE
    {
        FlowImage* img = context.read_pin<FlowImage>("In");
        if (img)
        {
            double min = 0.0;
            double max = 1.0;

            NumericObject* n = context.read_pin<NumericObject>("Min");
            if (n)
                min = n->as_float();

            n = context.read_pin<NumericObject>("Max");
            if (n)
                max = n->as_float();

            Image normalized = image::normalize_image(img->image(), min, max);
            context.write_pin("Out", new FlowImage(normalized));
        }
    }

    const char* title() const OVERRIDE
    {
        return "Normalize";
    }
    const char* category() const OVERRIDE
    {
        return "Image";
    }

};
IMPLEMENT_OBJECT(ImageNormalizeNode, "ImageNormalizeNode");




void flow_image_nodes::install()
{
    FlowSystem::get().install_template(new ImageLoadNode());
    FlowSystem::get().install_template(new ImageSaveNode());
    FlowSystem::get().install_template(new ImagePropertiesNode());
    FlowSystem::get().install_template(new ImageConvertNode());
    FlowSystem::get().install_template(new ImageNormalizeNode());
}
