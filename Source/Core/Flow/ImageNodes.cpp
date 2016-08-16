#include "Common.h"

#include "FlowContext.h"
#include "FlowImage.h"
#include "FlowNode.h"
#include "FlowPrimitives.h"
#include "FlowString.h"
#include "FlowSystem.h"
#include "FlowVector.h"
#include "Image/Image.h"
#include "Image/ITK.h"
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


void flow_image_nodes::install()
{
    FlowSystem::get().install_template(new ImageLoadNode());
    FlowSystem::get().install_template(new ImageSaveNode());
    FlowSystem::get().install_template(new ImagePropertiesNode());
}
