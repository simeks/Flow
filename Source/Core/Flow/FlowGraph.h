#ifndef __CORE_FLOW_GRAPH_H__
#define __CORE_FLOW_GRAPH_H__

#include "FlowNode.h"

class FlowGraph;
typedef std::shared_ptr<FlowGraph> FlowGraphPtr;


class CORE_API FlowGraph
{
public:
    FlowGraph();
    ~FlowGraph();

    void add_node(FlowNodePtr node);
    void remove_node(FlowNodePtr node);
    void clear();

    FlowNodePtr node(const Guid& id) const;
    const std::map<Guid, FlowNodePtr>& nodes() const;


private:
    std::map<Guid, FlowNodePtr> _nodes;

};

#endif // __CORE_FLOW_GRAPH_H__
