#ifndef __GPU_REGISTRATION_TRANSFORM_NODE_H__
#define __GPU_REGISTRATION_TRANSFORM_NODE_H__

#include <Core/Flow/FlowNode.h>

class GpuTransformNode : public FlowNode
{
    DECLARE_OBJECT(GpuTransformNode, FlowNode);
public:
    GpuTransformNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;
};

#endif // __GPU_REGISTRATION_TRANSFORM_NODE_H__
