#ifndef __CORE_FLOW_NODE_H__
#define __CORE_FLOW_NODE_H__

#include "FlowObject.h"
#include "FlowPin.h"

class FlowContext;
class FlowGraph;
class FlowNode;

typedef std::shared_ptr<FlowNode> FlowNodePtr;

class CORE_API FlowNode : public FlowObject
{
    DECLARE_OBJECT(FlowNode, FlowObject);

public:
    FlowNode();
    virtual ~FlowNode();

    virtual void run(FlowContext& context);

    const std::vector<FlowPinPtr>& pins() const;

    FlowPinPtr pin(int id) const;
    FlowPinPtr pin(const std::string& name) const;

    const Guid& node_id() const;
    void set_node_id(const Guid& id);
    void set_flow_graph(FlowGraph* graph);

    FlowNode(const FlowNode&);
    FlowNode& operator=(const FlowNode&);

    void add_pin(const std::string& name, FlowPin::Type pin_type);

    virtual std::string node_class() const;

    virtual const char* title() const;
    virtual const char* category() const;

protected:
    std::vector<FlowPinPtr> _pins;

    FlowGraph* _owner_graph;
    Guid _node_id;
};

#endif // __CORE_FLOW_NODE_H__
