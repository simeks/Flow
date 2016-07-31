#ifndef __CORE_TERMINAL_NODE_H__
#define __CORE_TERMINAL_NODE_H__

#include "FlowNode.h"

class CORE_API TerminalNode : public FlowNode
{
    DECLARE_OBJECT(TerminalNode, FlowNode);

public:
    TerminalNode();
    TerminalNode(FlowClass* input_type);
    virtual ~TerminalNode();

    virtual void run(FlowContext& context);

    const std::string& var_name() const;
    void set_var_name(const std::string& name);

    FlowObject* value() const;

    virtual std::string node_class() const OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;

    TerminalNode(const TerminalNode&);
    TerminalNode& operator=(const TerminalNode&);

private:
    std::string _var_name;
	FlowClass* _input_type;
    FlowObject* _value;
};

#endif // __CORE_TERMINAL_NODE_H__
