#ifndef __CORE_PYTHON_FLOW_CONTEXT_H__
#define __CORE_PYTHON_FLOW_CONTEXT_H__

#include "PythonWrapper.h"

class FlowContext;
namespace py_flow_module
{
	PyObject* create_context_type();
    PyObject* create_context(FlowContext* owner);
}

#endif // __CORE_PYTHON_FLOW_CONTEXT_H__
