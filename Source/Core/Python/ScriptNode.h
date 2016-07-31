#ifndef __CORE_PYTHON_SCRIPTNODE_H__
#define __CORE_PYTHON_SCRIPTNODE_H__

#include "PythonWrapper.h"
#include "Flow/FlowNode.h"

class ScriptNode : public FlowNode
{
    DECLARE_SCRIPT_OBJECT(ScriptNode, FlowNode);
public:
    ScriptNode();
    ~ScriptNode();

    virtual void run(FlowContext& context) OVERRIDE;

    virtual std::string node_class() const OVERRIDE;

    virtual const char* title() const OVERRIDE;
    virtual const char* category() const OVERRIDE;

	ScriptNode(const ScriptNode&);
	ScriptNode& operator=(const ScriptNode&);

protected:
    int script_object_init(PyObject* self, PyObject* args, PyObject* kwds) OVERRIDE;
};

#endif // __CORE_PYTHON_SCRIPTNODE_H__
