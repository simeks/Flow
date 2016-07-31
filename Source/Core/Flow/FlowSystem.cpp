#include "Common.h"

#include "DebugNodes.h"
#include "FlowCompiler.h"
#include "FlowContext.h"
#include "FlowNode.h"
#include "FlowPrimitives.h"
#include "FlowString.h"
#include "FlowSystem.h"
#include "ImageNodes.h"
#include "TerminalNode.h"

FlowSystem* FlowSystem::s_instance = nullptr;

FlowSystem& FlowSystem::get()
{
    assert(s_instance);
    return *s_instance;
}
void FlowSystem::create()
{
    if (!s_instance)
    {
        s_instance = new FlowSystem();
    }
}
void FlowSystem::destroy()
{
    if (s_instance)
    {
        delete s_instance;
        s_instance = nullptr;
    }
}

FlowSystem::FlowSystem()
{
}
FlowSystem::~FlowSystem()
{
    for (auto& node : _node_templates)
    {
        delete node;
    }
    _node_templates.clear();
}
void FlowSystem::initialize()
{
    flow_debug_nodes::install();
    flow_image_nodes::install();

    _node_templates.push_back(new TerminalNode(FlowInt::static_class()));
    _node_templates.push_back(new TerminalNode(FlowFloat::static_class()));
    _node_templates.push_back(new TerminalNode(FlowString::static_class()));
}
FlowGraphPtr FlowSystem::create_graph() const
{
    return std::make_shared<FlowGraph>();
}
void FlowSystem::install_template(FlowNode* node)
{
    _node_templates.push_back(node);
}
const std::vector<FlowNode*>& FlowSystem::node_templates() const
{
    return _node_templates;
}
FlowNode* FlowSystem::node_template(const std::string& node_class) const
{
    for (auto n : _node_templates)
    {
        if (n->node_class() == node_class)
            return n;
    }
    return nullptr;
}
