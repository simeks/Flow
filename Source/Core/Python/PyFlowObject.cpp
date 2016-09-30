#include "Common.h"

#include "Flow/FlowObject.h"
#include "PyFlowObject.h"
#include "PythonWrapper.h"
#include <structmember.h>

namespace py_flow_object
{
    const int kMaxObjectTypes = 1024;

    Type _object_types[kMaxObjectTypes];
    int _num_object_types = 0;
}

typedef struct {
    PyObject_HEAD
    FlowObject* owner;
} PyFlowObject;

void py_flow_object::type_dealloc(PyObject* self)
{
    self->ob_type->tp_free((PyObject*)self);
}
PyObject* py_flow_object::type_new(PyTypeObject* type, PyObject*, PyObject*)
{
    PyFlowObject *self;

    self = (PyFlowObject *)type->tp_alloc(type, 0);
    self->owner = nullptr;

    return (PyObject *)self;
}
int py_flow_object::type_init(PyObject* )
{
    return 0;

}
size_t py_flow_object::type_basicsize()
{
    return sizeof(PyFlowObject);
}

static PyObject* py_base_object_type_fn(PyFlowObject* self)
{
    if (self->owner)
    {
        return PyString_FromString(self->owner->get_class()->name());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

// Define base type

static PyMethodDef py_base_object_methods[] = {
    { "type", (PyCFunction)py_base_object_type_fn, METH_NOARGS, "Returns the type of the object." },
    { NULL }  /* Sentinel */
};

static PyTypeObject py_base_object_type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "flow.BaseObject",             /*tp_name*/
    (Py_ssize_t)py_flow_object::type_basicsize(),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)py_flow_object::type_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "FlowObject",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    py_base_object_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)py_flow_object::type_init,      /* tp_init */
    0,                         /* tp_alloc */
    py_flow_object::type_new,                 /* tp_new */
};

PyObject* py_flow_object::create_base_type()
{
    if (PyType_Ready(&py_base_object_type) < 0)
        return nullptr;

    Py_INCREF(&py_base_object_type);
    return (PyObject*)&py_base_object_type;
}
PyTypeObject* py_flow_object::base_type()
{
    return &py_base_object_type;
}
PyObject* py_flow_object::create_object(PyTypeObject* type, FlowObject* owner)
{
    PyFlowObject* obj = nullptr;

    PyObject* args = PyTuple_New(0);
    obj = (PyFlowObject*)type->tp_new(type, args, nullptr);
    Py_INCREF(obj);
    if (!obj)
    {
        console::print("Failed to create instance of python flow object:\n");
        PyErr_Print();
    }
    else
    {
        obj->owner = owner;
        type->tp_init((PyObject*)obj, args, nullptr);
    }
    Py_DECREF(args);

    return (PyObject*)obj;
}
FlowObject* py_flow_object::owner(PyObject* object)
{
    if (!PyType_IsSubtype((PyTypeObject*)PyObject_Type(object), &py_base_object_type))
        return nullptr;
    return ((PyFlowObject*)object)->owner;
}
void py_flow_object::set_owner(PyObject* object, FlowObject* owner)
{
    assert(object);
    ((PyFlowObject*)object)->owner = owner;
}

void py_flow_object::register_object_type(const char* name, PyObject* type, FlowClass* flow_type)
{
    _object_types[_num_object_types].name = name;
    _object_types[_num_object_types].type = type;
    _object_types[_num_object_types].flow_type = flow_type;
    ++_num_object_types;
}
py_flow_object::Type* py_flow_object::object_types()
{
    return _object_types;
}
int py_flow_object::num_object_types()
{
    return _num_object_types;
}
py_flow_object::Type* py_flow_object::find_object_type(const char* name)
{
    for (int i = 0; i < _num_object_types; ++i)
    {
        if (strcmp(name, _object_types[i].name) == 0)
            return &_object_types[i];
    }
    return nullptr;
}
