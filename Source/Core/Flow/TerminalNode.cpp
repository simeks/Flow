#include "Common.h"

#include "FlowContext.h"
#include "FlowPrimitives.h"
#include "FlowString.h"
#include "TerminalNode.h"

IMPLEMENT_OBJECT(TerminalNode, "TerminalNode");
TerminalNode::TerminalNode() 
    : _input_type(nullptr),
    _value(nullptr)
{
}
TerminalNode::TerminalNode(FlowClass* input_type) : 
    _input_type(input_type),
    _value(nullptr)
{
    _var_name = input_type->name();
    _var_name += "_param";

    add_pin("Out", FlowPin::Out);

    if (_input_type)
        _value = _input_type->create_object();
}
TerminalNode::~TerminalNode()
{
    delete _value;
}
void TerminalNode::run(FlowContext& context)
{
    // TODO: Cache values?
    auto it = context.env_vars.find(_var_name);
    if (it != context.env_vars.end())
    {
        const std::string& value = it->second;
        if (_input_type == FlowInt::static_class())
        {
            FlowInt* obj = (FlowInt*)FlowInt::create_object();
            obj->set(std::stoi(value));
            context.write_pin("Out", obj);
            delete obj;
        }
        else if (_input_type == FlowFloat::static_class())
        {
            FlowFloat* obj = (FlowFloat*)FlowFloat::create_object();
            obj->set(std::stof(value));
            context.write_pin("Out", obj);
            delete obj;
        }
        else if (_input_type == FlowString::static_class())
        {
            FlowString* obj = (FlowString*)FlowString::create_object();
            obj->set(value);
            context.write_pin("Out", obj);
            delete obj;
        }
        else
        {
            assert(false);
        }
    }
    else
    {
        // Write default value
        context.write_pin("Out", _value);
    }
}
const std::string& TerminalNode::var_name() const
{
    return _var_name;
}
void TerminalNode::set_var_name(const std::string& name)
{
    _var_name = name;
}
FlowObject* TerminalNode::value() const
{
    return _value;
}

std::string TerminalNode::node_class() const
{
    std::string node_class = get_class()->name();
    node_class += ":";
    node_class += _input_type->name();
    return node_class;
}
const char* TerminalNode::title() const
{
    return _input_type->name();
}
const char* TerminalNode::category() const
{
    return "Parameter";
}

TerminalNode::TerminalNode(const TerminalNode& other) : FlowNode(other)
{
    _input_type = other._input_type;
    if (other._value)
        _value = other._value->clone();
}
TerminalNode& TerminalNode::operator=(const TerminalNode& other)
{
    FlowNode::operator=(other);

    _var_name = other._var_name;
    _input_type = other._input_type;
    if (other._value)
        _value = other._value->clone();

    return *this;
}
