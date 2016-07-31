#ifndef __PY_FLOW_NODE_H__
#define __PY_FLOW_NODE_H__

#include "PythonWrapper.h"

class ScriptNode;
namespace py_flow_node
{
    PyObject* copy_node(PyObject* src, ScriptNode* node);

	void run_node(PyObject* obj, PyObject* context);
}

#endif // __PY_FLOW_NODE_H__
