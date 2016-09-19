#include <Core/Common.h>
#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Flow/FlowPrimitives.h>

#include "ImageToWorld.h"

namespace 
{
    template<typename TImage>
    TImage transform_image_to_world(const TImage& source, int axis, int idx)
    {
        if (axis == 2) // Z-axis
        {
            Vec3i dst_size(
                (int)ceil(source.size()[0] * source.spacing()[0]), // X
                (int)ceil(source.size()[1] * source.spacing()[1]), // Y
                1);

            TImage dst(2, dst_size);
            dst.set_spacing(Vec3i(1, 1, 1));
            dst.set_origin(source.origin());

#pragma omp parallel for
            for (int y = 0; y < dst_size.y; ++y)
            {
                for (int x = 0; x < dst_size.x; ++x)
                {
                    dst(x, y, 0) = source.linear_at(x / source.spacing()[0], y / source.spacing()[1], idx, image::Border_Replicate);
                }
            }
            return dst;
        }
        else if (axis == 1) // Y-axis
        {
            Vec3i dst_size(
                (int)ceil(source.size()[0] * source.spacing()[0]), // X
                (int)ceil(source.size()[2] * source.spacing()[2]), // Z
                1);

            TImage dst(2, dst_size);
            dst.set_spacing(Vec3i(1, 1, 1));
            dst.set_origin(source.origin());

#pragma omp parallel for
            for (int y = 0; y < dst_size.y; ++y)
            {
                for (int x = 0; x < dst_size.x; ++x)
                {
                    // Flip the y-axis in the resulting image
                    dst(x, dst_size.y - y - 1, 0) = source.linear_at(x / source.spacing()[0], idx, y / source.spacing()[2], image::Border_Replicate);
                }
            }
            return dst;
        }
        else // X-axis
        {
            Vec3i dst_size(
                (int)ceil(source.size()[1] * source.spacing()[1]), // Y
                (int)ceil(source.size()[2] * source.spacing()[2]), // Z
                1);

            TImage dst(2, dst_size);
            dst.set_spacing(Vec3i(1, 1, 1));
            dst.set_origin(source.origin());

#pragma omp parallel for
            for (int y = 0; y < dst_size.y; ++y)
            {
                for (int x = 0; x < dst_size.x; ++x)
                {
                    // Flip the y-axis in the resulting image
                    dst(x, dst_size.y - y - 1, 0) = source.linear_at(idx, x / source.spacing()[1], y / source.spacing()[2], image::Border_Replicate);
                }
            }
            return dst;
        }
    }
}

IMPLEMENT_OBJECT(ImageSliceToWorldNode, "ImageSliceToWorldNode");

ImageSliceToWorldNode::ImageSliceToWorldNode()
{
    add_pin("In", FlowPin::In);
    add_pin("Index", FlowPin::In);
    add_pin("Axis", FlowPin::In);
    add_pin("Out", FlowPin::Out);
}
ImageSliceToWorldNode::~ImageSliceToWorldNode()
{
}
void ImageSliceToWorldNode::run(FlowContext& context)
{
    FlowImage* in = context.read_pin<FlowImage>("In");
    FlowInt* idx = context.read_pin<FlowInt>("Index");
    FlowInt* axis = context.read_pin<FlowInt>("Axis");
    if (in)
    {
        FlowImage* out = nullptr;

        int slice_index = 126;
        if (idx)
            slice_index = idx->get();

        int slice_axis = 2;
        if (axis)
            slice_axis = axis->get();

        if (in->pixel_type() == image::PixelType_Float32)
        {
            Image img = transform_image_to_world<ImageFloat32>(in->image(), slice_axis, slice_index);
            out = new FlowImage(img);
        }
        else if (in->pixel_type() == image::PixelType_Float64)
        {
            Image img = transform_image_to_world<ImageFloat64>(in->image(), slice_axis, slice_index);
            out = new FlowImage(img);
        }
        else if (in->pixel_type() == image::PixelType_Vec4u8)
        {
            // Perform the transformation on floating-point values and then convert back to RGBA32
            ImageRGBA32 img = transform_image_to_world<ImageColorf>(in->image(), slice_axis, slice_index);
            out = new FlowImage(img);
        }
        else if (in->pixel_type() == image::PixelType_Vec4f)
        {
            Image img = transform_image_to_world<ImageColorf>(in->image(), slice_axis, slice_index);
            out = new FlowImage(img);
        }
        else if (in->pixel_type() == image::PixelType_Vec4d)
        {
            Image img = transform_image_to_world<ImageColord>(in->image(), slice_axis, slice_index);
            out = new FlowImage(img);
        }
        else
        {
            assert(false);
        }
        context.write_pin("Out", out);
    }

}
const char* ImageSliceToWorldNode::title() const
{
    return "ImageSliceToWorld";
}
const char* ImageSliceToWorldNode::category() const
{
    return "Image";
}