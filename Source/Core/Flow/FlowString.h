#ifndef __CORE_FLOW_STRING_H__
#define __CORE_FLOW_STRING_H__

#include "FlowObject.h"

class CORE_API FlowString : public FlowObject
{
    DECLARE_OBJECT(FlowString, FlowObject);
public:
    FlowString();
    FlowString(const std::string& value);
    ~FlowString();

    const std::string& get() const;
    void set(const std::string& value);

private:
    std::string _value;
private:
    virtual PyObject* py_create_object();
};

#endif // __CORE_FLOW_STRING_H__
