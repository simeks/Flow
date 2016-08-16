#ifndef __REGISTRATION_TRANSFORM_SLICE_NODE_H__
#define __REGISTRATION_TRANSFORM_SLICE_NODE_H__

#include <Core/Flow/FlowNode.h>

class TransformSliceNode : public FlowNode
{
    DECLARE_OBJECT(TransformSliceNode, FlowNode);
public:
    TransformSliceNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;
};

#endif // __REGISTRATION_TRANSFORM_SLICE_NODE_H__
