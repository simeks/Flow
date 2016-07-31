#ifndef __CORE_FLOW_PIN_H__
#define __CORE_FLOW_PIN_H__

#include "FlowObject.h"

class FlowNode;
class FlowPin;

typedef std::shared_ptr<FlowPin> FlowPinPtr;

class CORE_API FlowPin : public FlowObject
{
    DECLARE_OBJECT(FlowPin, FlowObject);

public:
    enum Type
    {
        In,
        Out,
        Unknown
    };

    FlowPin();
    FlowPin(const std::string& name,
            Type pin_type,
            FlowNode* owner, int id);
    virtual ~FlowPin();

    void link_to(FlowPin* pin);
    void unlink(FlowPin* pin);
    void unlink_all();
    const std::vector<FlowPin*> links() const;
    bool is_linked() const;
    
    Type pin_type() const;
    const std::string& name() const;

    FlowNode* owner() const;
    int pin_id() const;

private:
    Type _pin_type;
    std::string _name;

    std::vector<FlowPin*> _links;

    FlowNode* _owner;
    int _id;
};

#endif // __CORE_FLOW_PIN_H__
