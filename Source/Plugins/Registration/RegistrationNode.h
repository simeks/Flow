#ifndef __REGISTRATION_NODE_H__
#define __REGISTRATION_NODE_H__

#include <Core/Flow/FlowNode.h>

class Image;
class RegistrationNode : public FlowNode
{
    DECLARE_OBJECT(RegistrationNode, FlowNode);
public:
    RegistrationNode();
    virtual ~RegistrationNode();

    virtual void run(FlowContext& context) OVERRIDE;

    virtual const char* title() const OVERRIDE;
    virtual const char* category() const OVERRIDE;
};


#endif // __REGISTRATION_NODE_H__
