#include "Common.h"

#include "FlowNode.h"


IMPLEMENT_OBJECT(FlowNode, "FlowNode");
FlowNode::FlowNode() :
    _owner_graph(nullptr)
{
}
FlowNode::~FlowNode()
{
    _pins.clear();
}
void FlowNode::run(FlowContext& )
{
}
const std::vector<FlowPinPtr>& FlowNode::pins() const
{
    return _pins;
}
FlowPinPtr FlowNode::pin(int id) const
{
    return _pins[id];
}
FlowPinPtr FlowNode::pin(const std::string& name) const
{
    for (auto& p : _pins)
    {
        if (p->name() == name)
        {
            return p;
        }
    }
    return nullptr;
}

const Guid& FlowNode::node_id() const
{
    return _node_id;
}
void FlowNode::set_node_id(const Guid& id)
{
    _node_id = id;
}
void FlowNode::set_flow_graph(FlowGraph* graph)
{
    _owner_graph = graph;
}

void FlowNode::add_pin(const std::string& name, FlowPin::Type pin_type)
{
    int id = (int)_pins.size();
    _pins.push_back(std::make_shared<FlowPin>(name, pin_type, this, id));
}
std::string FlowNode::node_class() const
{
   return get_class()->name();
}
const char* FlowNode::title() const
{
    return get_class()->name();
}
const char* FlowNode::category() const
{
    return "";
}
FlowNode::FlowNode(const FlowNode& other) : FlowObject(other)
{
    for (auto& pin : other._pins)
    {
        add_pin(pin->name(), pin->pin_type());
    }
    _owner_graph = other._owner_graph;
    _node_id = other._node_id;
}
FlowNode& FlowNode::operator=(const FlowNode& other)
{
    FlowObject::operator=(other);

    _pins.clear();
    for (auto& pin : other._pins)
    {
        add_pin(pin->name(), pin->pin_type());
    }
    _owner_graph = other._owner_graph;
    _node_id = other._node_id;

    return *this;
}
