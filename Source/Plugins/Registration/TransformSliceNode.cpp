#include <Core/Common.h>

#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Flow/FlowPrimitives.h>
#include <Core/Image/Image.h>

#include "TransformSliceNode.h"

namespace transform
{
    template<typename TImage>
    TImage transform_slice(const TImage& source, const ImageVec3d& deformation, int zindex)
    {
        TImage result(source.ndims(), source.size());
        result.set_spacing(source.spacing());
        result.set_origin(source.origin());
        Vec3i size = result.size();
        Vec3d spacing = deformation.spacing();

#pragma omp parallel for
        for (int y = 0; y < size.y; ++y)
        {
            for (int x = 0; x < size.x; ++x)
            {
                Vec3d d = deformation.linear_at(x / spacing.x, y / spacing.y, zindex);
                d.x = d.x * spacing.x;
                d.y = d.y * spacing.y;
                result(x, y, 0) = source.linear_at(x + d.x, y + d.y, 0);
            }
        }
        return result;
    }
}

IMPLEMENT_OBJECT(TransformSliceNode, "TransformSliceNode");

TransformSliceNode::TransformSliceNode()
{
    add_pin("Source", FlowPin::In);
    add_pin("Deformation", FlowPin::In);
    add_pin("ZIndex", FlowPin::In);
    add_pin("Result", FlowPin::Out);
}
void TransformSliceNode::run(FlowContext& context)
{
    FlowImage* image_obj = context.read_pin<FlowImage>("Source");
    FlowImage* deformation_obj = context.read_pin<FlowImage>("Deformation");
    FlowInt* slice_index = context.read_pin<FlowInt>("ZIndex");
    if (image_obj && deformation_obj && slice_index)
    {
        ImageVec3d deformation = deformation_obj->image();

        int zindex = slice_index->get();

        FlowImage* result = nullptr;
        if (image_obj->pixel_type() == image::PixelType_UInt8)
        {
            Image img = transform::transform_slice<ImageUInt8>(image_obj->image(), deformation, zindex);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Float32)
        {
            Image img = transform::transform_slice<ImageFloat32>(image_obj->image(), deformation, zindex);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Float64)
        {
            Image img = transform::transform_slice<ImageFloat64>(image_obj->image(), deformation, zindex);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Vec4u8)
        {
            ImageRGBA32 img = transform::transform_slice<ImageColorf>(image_obj->image(), deformation, zindex);
            result = new FlowImage(img);
        }
        else if (image_obj->pixel_type() == image::PixelType_Vec4f)
        {
            Image img = transform::transform_slice<ImageColorf>(image_obj->image(), deformation, zindex);
            result = new FlowImage(img);
        }
        else
        {
            assert(false);
        }
        context.write_pin("Result", result);
    }
}
const char* TransformSliceNode::title() const
{
    return "TransformSlice";
}
const char* TransformSliceNode::category() const
{
    return "Registration";
}
