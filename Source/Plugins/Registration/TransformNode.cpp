#include <Core/Common.h>

#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Image/Image.h>

#include "TransformNode.h"

namespace transform
{
    template<typename TImage>
    TImage transform_image(const TImage& source, const ImageVec3d& deformation)
    {
        assert(source.size() == deformation.size());

        TImage result(source.ndims(), source.size());
        result.set_spacing(source.spacing());
        result.set_origin(source.origin());
        Vec3i size = result.size();
#pragma omp parallel for
        for (int z = 0; z < size.z; ++z)
        {
            for (int y = 0; y < size.y; ++y)
            {
                for (int x = 0; x < size.x; ++x)
                {
                    result(x, y, z) = source.linear_at(Vec3d(x, y, z) + deformation(x, y, z));
                }
            }
        }
        return result;
    }
}

IMPLEMENT_OBJECT(TransformNode, "TransformNode");

TransformNode::TransformNode()
{
    add_pin("Source", FlowPin::In);
    add_pin("Deformation", FlowPin::In);
    add_pin("Result", FlowPin::Out);
}
void TransformNode::run(FlowContext& context)
{
    FlowImage* image_obj = context.read_pin<FlowImage>("Source");
    FlowImage* deformation_obj = context.read_pin<FlowImage>("Deformation");
    if (image_obj && deformation_obj) 
    {
        ImageVec3d deformation = deformation_obj->image();
        assert(deformation.size() == image_obj->size());

        FlowImage* result = nullptr;
        if (image_obj->pixel_type() == image::PixelType_Float32)
        {
            Image img = transform::transform_image<ImageFloat32>(image_obj->image(), deformation);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Float64)
        {
            Image img = transform::transform_image<ImageFloat64>(image_obj->image(), deformation);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Vec4u8)
        {
            ImageRGBA32 img = transform::transform_image<ImageColorf>(image_obj->image(), deformation);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Vec4f)
        {
            Image img = transform::transform_image<ImageColorf>(image_obj->image(), deformation);
            result = new FlowImage(img);
        }
        else
        {
            assert(false);
        }
        context.write_pin("Result", result);
    }
}
const char* TransformNode::title() const
{
    return "Transform";
}
const char* TransformNode::category() const
{
    return "Registration";
}
