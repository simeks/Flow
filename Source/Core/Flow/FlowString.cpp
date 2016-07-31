#include "Common.h"

#include "FlowString.h"

#include "Python/PyFlowModule.h"
#include "Python/PyFlowObject.h"
#include "Python/PythonWrapper.h"


IMPLEMENT_OBJECT(FlowString, "String");

FlowString::FlowString() : _value("")
{
}
FlowString::FlowString(const std::string& value) : _value(value)
{
}
FlowString::~FlowString()
{
}

const std::string& FlowString::get() const
{
    return _value;
}
void FlowString::set(const std::string& value)
{
    _value = value;
}
PyObject* FlowString::py_create_object()
{
    return PyString_FromString(_value.c_str());
}

