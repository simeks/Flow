#include "Common.h"

#include "FlowContext.h"
#include "FlowPrimitives.h"
#include "FlowString.h"
#include "Python/PyFlowContext.h"

FlowContext::FlowContext() : active_node(nullptr)
{
    script_object = py_flow_module::create_context(this);
}
FlowContext::~FlowContext()
{
    Py_DECREF(script_object);
}

FlowObject* FlowContext::read_pin(int id) const
{
    if (active_node)
        return read_pin(active_node->pin(id));
    else
        return nullptr;
}
FlowObject* FlowContext::read_pin(const std::string& pin_name) const
{
    if (active_node)
        return read_pin(active_node->pin(pin_name));
    else
        return nullptr;
}
FlowObject* FlowContext::read_pin(FlowPinPtr pin) const
{
    assert(pin->pin_type() == FlowPin::In);
    if (pin->pin_type() == FlowPin::In)
    {
        auto it = pin_data.find(pin.get());
        if (it != pin_data.end())
        {
            return it->second;
        }
    }
    return nullptr;
}
int64_t FlowContext::read_int(const std::string& pin_name)
{
    NumericObject* obj = read_pin<NumericObject>(pin_name);
    if (!obj)
        FATAL_ERROR("Failed to read pin '%s'.", pin_name.c_str());

    return obj->as_int();
}
double FlowContext::read_float(const std::string& pin_name)
{
    NumericObject* obj = read_pin<NumericObject>(pin_name);
    if (!obj)
        FATAL_ERROR("Failed to read pin '%s'.", pin_name.c_str());

    return obj->as_float();
}
const std::string& FlowContext::read_string(const std::string& pin_name)
{
    FlowString* obj = read_pin<FlowString>(pin_name);
    if (!obj)
        FATAL_ERROR("Failed to read pin '%s'.", pin_name.c_str());

    return obj->get();
}
void FlowContext::write_pin(int id, FlowObject* data)
{
    if (active_node)
    {
        FlowPinPtr pin = active_node->pin(id);
        write_pin(pin, data);
    }
}
void FlowContext::write_pin(const std::string& pin_name, FlowObject* data)
{
    if (active_node)
    {
        FlowPinPtr pin = active_node->pin(pin_name);
        write_pin(pin, data);
    }
}
void FlowContext::write_pin(FlowPinPtr pin, FlowObject* data)
{
    assert(pin->pin_type() == FlowPin::Out);
    if (pin->pin_type() == FlowPin::Out)
    {
        for (auto& target_pin : pin->links())
        {
            auto it = pin_data.find(target_pin);
            if (it != pin_data.end())
            {
                FlowObject* copy = data->clone();
                objects.push_back(copy);
                it->second = copy;
            }
        }
    }
}
std::string FlowContext::env_var(const std::string& key) const
{
    auto it = env_vars.find(key);
    if (it != env_vars.end())
        return it->second;
    return "";
}
void FlowContext::allocate_context(FlowGraphPtr graph)
{
    for (auto& node : graph->nodes())
    {
        for (auto& pin : node.second->pins())
        {
            if (pin->pin_type() == FlowPin::In)
            {
                pin_data[pin.get()] = nullptr;
            }
        }
    }
}
void FlowContext::run()
{
    for (auto& n : execution_order)
    {
        active_node = n;
        n->run(*this);
    }
}
void FlowContext::clean_up()
{
    for (FlowObject* obj : objects)
    {
        delete obj;
    }
    objects.clear();
    active_node = nullptr;
}

