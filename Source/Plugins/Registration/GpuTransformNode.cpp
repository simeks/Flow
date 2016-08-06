#include <Core/Common.h>

#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Image/Image.h>

#include "GpuTransformNode.h"
#include "Cuda/Transform.h"

IMPLEMENT_OBJECT(GpuTransformNode, "GpuTransformNode");

GpuTransformNode::GpuTransformNode()
{
    add_pin("Source", FlowPin::In);
    add_pin("Deformation", FlowPin::In);
    add_pin("Result", FlowPin::Out);
}
void GpuTransformNode::run(FlowContext& context)
{
    FlowImage* image_obj = context.read_pin<FlowImage>("Source");
    FlowImage* deformation_obj = context.read_pin<FlowImage>("Deformation");
    if (image_obj && deformation_obj)
    {
        ImageVec3d deformation = deformation_obj->image();
        assert(deformation.size() == image_obj->size());

        Image result = cuda::transform_image(image_obj->image(), deformation);
        context.write_pin("Result", new FlowImage(result));
    }
}
const char* GpuTransformNode::title() const
{
    return "Transform (CUDA)";
}
const char* GpuTransformNode::category() const
{
    return "Registration/CUDA";
}

