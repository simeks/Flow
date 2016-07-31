#include <Core/Common.h>

#include "FlowCompiler.h"
#include "FlowContext.h"
#include "FlowGraph.h"

FlowCompiler::FlowCompiler()
{

}
FlowCompiler::~FlowCompiler()
{

}
bool FlowCompiler::compile(FlowContext& context, FlowGraphPtr graph)
{
    context.allocate_context(graph);
    context.execution_order.clear();
    return create_execution_list(graph, context.execution_order);

}
bool FlowCompiler::create_execution_list(FlowGraphPtr graph, std::vector<FlowNode*>& execution_list)
{
    std::vector<FlowNode*> terminal_nodes;
    std::map<FlowNode*, int> num_incoming_edges;
    int total_incoming_edges_left = 0;

    for (auto& node_it : graph->nodes())
    {
        FlowNodePtr node = node_it.second;

        int incoming_edges = 0;
        for (auto& pin : node->pins())
        {
            if (pin->pin_type() == FlowPin::In && !pin->links().empty())
            {
                ++incoming_edges;
            }
        }
        num_incoming_edges[node.get()] = incoming_edges;
        total_incoming_edges_left += incoming_edges;

        if (incoming_edges == 0)
        {
            terminal_nodes.push_back(node.get());
        }
    }

    while (!terminal_nodes.empty())
    {
        FlowNode* node = terminal_nodes.back();
        terminal_nodes.pop_back();

        execution_list.push_back(node);

        for (auto& pin : node->pins())
        {
            if (pin->pin_type() == FlowPin::Out)
            {
                for (auto& link : pin->links())
                {
                    auto incoming_edges = num_incoming_edges.find(link->owner());
                    if (incoming_edges != num_incoming_edges.end())
                    {
                        int& num = incoming_edges->second;
                        --num;
                        --total_incoming_edges_left;

                        if (num <= 0)
                        {
                            terminal_nodes.push_back(link->owner());
                        }
                    }
                }
            }
        }
    }

    if (total_incoming_edges_left)
    {
        console::error("Failed to compile graph: Cycles detected.");
        execution_list.clear();
        return false;
    }
    return true;
}

