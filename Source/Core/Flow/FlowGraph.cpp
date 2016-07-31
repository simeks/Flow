#include "Common.h"

#include "FlowNode.h"
#include "FlowGraph.h"

FlowGraph::FlowGraph()
{
}
FlowGraph::~FlowGraph()
{
}

void FlowGraph::add_node(FlowNodePtr node)
{
    assert(node);
    assert(_nodes.find(node->node_id()) == _nodes.end()); // Collision detected: Node ID already exists in graph

    if (!node->node_id().is_valid())
    {
        // Node have no ID, assign a new one
        node->set_node_id(guid::create_guid());
    }
    _nodes[node->node_id()] = node;

    node->set_flow_graph(this);
    
}
void FlowGraph::remove_node(FlowNodePtr node)
{
    assert(node);
    for (auto it = _nodes.begin(); it != _nodes.end(); ++it)
    {
        if (it->second == node)
        {
            _nodes.erase(it);
            break;
        }
    }
    node->set_flow_graph(nullptr);
}
void FlowGraph::clear()
{
    _nodes.clear();
}
FlowNodePtr FlowGraph::node(const Guid& id) const
{
    assert(id.is_valid());
    auto it = _nodes.find(id);
    assert(it != _nodes.end());
    return it->second;
}
const std::map<Guid, FlowNodePtr>& FlowGraph::nodes() const
{
    return _nodes;
}
