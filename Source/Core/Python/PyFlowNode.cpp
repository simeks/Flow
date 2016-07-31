#include "Common.h"
#include "Flow/FlowPrimitives.h"
#include "Flow/FlowString.h"
#include "PyFlowNode.h"
#include "PyFlowObject.h"
#include "ScriptNode.h"
#include "PythonWrapper.h"

#include <structmember.h>


PyObject* py_flow_node::copy_node(PyObject* src, ScriptNode* node)
{
    PyObject* obj = nullptr;
    PyTypeObject* type = (PyTypeObject*)PyObject_Type(src);
    if (src)
    {
        PyObject* args = Py_BuildValue("(O)", src);
        obj = type->tp_new(type, args, nullptr);
        if (!obj)
        {
            console::print("Failed to create instance of python node:\n");
            PyErr_Print();
        }
        else
        {
            py_flow_object::set_owner(obj, node);
            type->tp_init((PyObject*)obj, args, nullptr);
        }
        Py_DECREF(args);
    }

    return (PyObject*)obj;
}

void py_flow_node::run_node(PyObject* obj, PyObject* context)
{
    Py_INCREF(context);
    PyObject* ret = PyObject_CallMethod((PyObject*)obj, "run", "O", context);
    if (!ret)
    {
        PyErr_Print();
    }
}
