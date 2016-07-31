#ifndef __CORE_FLOW_SYSTEM_H__
#define __CORE_FLOW_SYSTEM_H__

#include "FlowGraph.h"

class FlowFunctionLibrary;
class FlowNode;
class FlowNodeClass;

class CORE_API FlowSystem
{
public:
    FlowSystem();
    ~FlowSystem();

    void initialize();

    FlowGraphPtr create_graph() const;

    void install_template(FlowNode* node);

    const std::vector<FlowNode*>& node_templates() const;
    FlowNode* node_template(const std::string& node_class) const;

    static FlowSystem& get();
    static void create();
    static void destroy();

private:
    static FlowSystem* s_instance;

    std::vector<FlowNode*> _node_templates;

};

#endif // __CORE_FLOW_SYSTEM_H__
