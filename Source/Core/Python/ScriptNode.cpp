#include "Common.h"

#include "Flow/FlowContext.h"
#include "Flow/FlowPrimitives.h"
#include "Flow/FlowString.h"
#include "PyFlowNode.h"
#include "Python/PyFlowModule.h"
#include "Python/PyFlowObject.h"
#include "Python/PythonWrapper.h"
#include "ScriptNode.h"

static PyObject *
py_ScriptNode_add_pin(PyObject* self, PyObject* args, PyObject *)
{
    PyObject* ret = nullptr;

    ScriptNode* object = object_cast<ScriptNode>(py_flow_object::owner(self));
    if (object)
    {
        char* pin_name;
        char* pin_type;
        if (PyArg_ParseTuple(args, "ss", &pin_name, &pin_type))
        {
            for (int i = 0; pin_type[i]; ++i) pin_type[i] = (char)tolower(pin_type[i]);

            FlowPin::Type pin_t = FlowPin::Unknown;
            if (strcmp(pin_type, "in") == 0)
            {
                pin_t = FlowPin::In;
            }
            else if (strcmp(pin_type, "out") == 0)
            {
                pin_t = FlowPin::Out;
            }
            else
            {
                PyErr_SetString(PyExc_AttributeError, "Invalid pin type: expecting either 'In' or 'Out'.");
                return nullptr;
            }

            object->add_pin(pin_name, pin_t);

            Py_INCREF(Py_None);
            ret = Py_None;
        }
    }
    return ret;
}

static PyObject *
py_ScriptNode_is_pin_linked(PyObject* self, PyObject* args, PyObject *)
{
    ScriptNode* object = object_cast<ScriptNode>(py_flow_object::owner(self));
    if (object)
    {
        const char* pin_name;
        if (PyArg_ParseTuple(args, "s", &pin_name))
        {
            if (object->pin(pin_name)->is_linked())
            {
                Py_RETURN_TRUE;
            }
            else
            {
                Py_RETURN_FALSE;
            }
        }
    }
    return nullptr;
}



static PyMethodDef py_ScriptNode_methods[] = {
    { "add_pin", (PyCFunction)py_ScriptNode_add_pin, METH_VARARGS, "Adds a pin to the node : add_pin(name, pin_type)." },
    { "is_pin_linked", (PyCFunction)py_ScriptNode_is_pin_linked, METH_VARARGS, "Checks whether the specified pin is linked." },
    { NULL }  /* Sentinel */
};

IMPLEMENT_SCRIPT_OBJECT(ScriptNode, "ScriptNode", "Node", py_ScriptNode_methods);

ScriptNode::ScriptNode()
{
}
ScriptNode::~ScriptNode()
{
}
void ScriptNode::run(FlowContext& context)
{
    if (_py_object)
    {
        py_flow_node::run_node(_py_object, context.script_object);
    }
}

std::string ScriptNode::node_class() const
{
    PyObject* type = PyObject_Type(_py_object);
    PyObject* module_name = PyObject_GetAttrString(type, "__module__");

    PyObject* class_name = nullptr;
    if (PyObject_HasAttrString(_py_object, "class_name"))
        class_name = PyObject_GetAttrString(_py_object, "class_name");
    else
        class_name = PyObject_GetAttrString(type, "__name__");

    std::string node_class = get_class()->name();
    node_class += ":";
    node_class += PyString_AsString(module_name);
    node_class += ".";
    node_class += PyString_AsString(class_name);

    return node_class;
}
const char* ScriptNode::title() const
{
    if (_py_object)
    {
        if (PyObject* obj = PyObject_GetAttrString(_py_object, "title"))
        {
            return PyString_AsString(obj);
        }
        else
        {
            PyObject* type = PyObject_Type(_py_object);
            PyObject* class_name = PyObject_GetAttrString(type, "__name__");
            return PyString_AsString(class_name);
        }
    }
    return "";
}
const char* ScriptNode::category() const
{
    if (_py_object)
    {
        if (PyObject* obj = PyObject_GetAttrString(_py_object, "category"))
        {
            return PyString_AsString(obj);
        }
    }
    return "";
}

ScriptNode::ScriptNode(const ScriptNode& other) : FlowNode(other)
{
    _py_object = py_flow_node::copy_node(other._py_object, this);
    Py_XINCREF(_py_object);
}
ScriptNode& ScriptNode::operator=(const ScriptNode& other)
{
	FlowNode::operator=(other);
    _py_object = py_flow_node::copy_node(other._py_object, this);
    Py_XINCREF(_py_object);

	return *this;
}

int ScriptNode::script_object_init(PyObject* , PyObject*, PyObject*)
{
    return 0;
}
