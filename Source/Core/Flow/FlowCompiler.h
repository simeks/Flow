#ifndef __CORE_FLOW_COMPILER_H__
#define __CORE_FLOW_COMPILER_H__

#include "FlowGraph.h"
#include "FlowNode.h"

class FlowContext;
class CORE_API FlowCompiler
{
public:
    FlowCompiler();
    ~FlowCompiler();

    bool compile(FlowContext& context, FlowGraphPtr graph);

private:
    bool create_execution_list(FlowGraphPtr graph, std::vector<FlowNode*>& execution_list);

};

#endif // __CORE_FLOW_COMPILER_H__
