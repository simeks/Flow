#include "Common.h"
#include "PyFlowContext.h"
#include "PyFlowModule.h"
#include "PyFlowNode.h"
#include "PyFlowObject.h"
#include "ScriptNode.h"
#include "Flow/FlowSystem.h"

#include "PythonWrapper.h"

namespace py_flow_module
{
    const int kMaxNumTypes = 1024;
    struct Module
    {
        PyObject* module = nullptr;
    } g_module;

    PyObject* install_template(PyObject*, PyObject* args)
    {
        PyObject* ret = nullptr;
        PyObject* obj = nullptr;
        if (PyArg_ParseTuple(args, "O:install_template", &obj))
        {
            if (!PyType_IsSubtype((PyTypeObject*)PyObject_Type(obj), ScriptNode::py_type()))
            {
                PyErr_SetString(PyExc_TypeError, "parameter must be a object inheriting flow.Node.");
                return NULL;
            }
            Py_INCREF(obj);

            ScriptNode* node = object_cast<ScriptNode>(py_flow_object::owner(obj));
            FlowSystem::get().install_template(node);

            Py_INCREF(Py_None);
            ret = Py_None;
        }
        return ret;
    }
}

void py_flow_module::init_module()
{
    static PyMethodDef module_methods[] =
    {
        { "install_template", install_template, METH_VARARGS, "Installs a node template." },
        { NULL, NULL, 0, NULL }
    };
    g_module.module = Py_InitModule("flow", module_methods);

    PyModule_AddObject(g_module.module, "Context", create_context_type());
    PyModule_AddObject(g_module.module, "BaseObject", py_flow_object::create_base_type());

    for (int i = 0; i < py_flow_object::num_object_types(); ++i)
    {
        const py_flow_object::Type& type = py_flow_object::object_types()[i];
        if (PyType_Ready((PyTypeObject*)type.type) < 0)
            continue;
        Py_INCREF(type.type);
        PyModule_AddObject(g_module.module, type.name, type.type);
    }
}

