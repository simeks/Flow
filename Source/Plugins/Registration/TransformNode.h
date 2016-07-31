#ifndef __REGISTRATION_TRANSFORM_NODE_H__
#define __REGISTRATION_TRANSFORM_NODE_H__

#include <Core/Flow/FlowNode.h>

class TransformNode : public FlowNode
{
    DECLARE_OBJECT(TransformNode, FlowNode);
public:
    TransformNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;
};

#endif // __REGISTRATION_TRANSFORM_NODE_H__
