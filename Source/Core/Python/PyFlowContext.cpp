#include "Common.h"

#include "Flow/FlowContext.h"
#include "Flow/FlowString.h"
#include "Flow/FlowPrimitives.h"
#include "PyFlowContext.h"
#include "PyFlowObject.h"

#include <structmember.h>

typedef struct {
    PyObject_HEAD
    FlowContext* owner;
} PyFlowContext;

static void
PyFlowContext_dealloc(PyFlowContext* self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
PyFlowContext_new(PyTypeObject *type, PyObject *, PyObject *)
{
    PyFlowContext *self;

    self = (PyFlowContext *)type->tp_alloc(type, 0);
    self->owner = nullptr;

    return (PyObject *)self;
}

static int
PyFlowContext_init(PyFlowContext* , PyObject* , PyObject *)
{
    return 0;
}

static PyObject *
PyFlowContext_read_pin(PyFlowContext* self, PyObject* args, PyObject *)
{
    PyObject* ret = nullptr;

    char* pin_name;
    if (PyArg_ParseTuple(args, "s", &pin_name))
    {
        if (self->owner)
        {
            FlowObject* obj = self->owner->read_pin(pin_name);
            if(obj)
            {
                PyObject* py_obj = obj->script_object();
                Py_INCREF(py_obj);
                ret = py_obj;
            }
            else
            {
                Py_RETURN_NONE;
            }
        }
        else
        {
            Py_RETURN_NONE;
        }
    }
    return ret;
}

static PyObject *
PyFlowContext_write_pin(PyFlowContext* self, PyObject* args, PyObject *)
{
    PyObject* ret = nullptr;

    char* pin_name;
    PyObject* obj;
    if (PyArg_ParseTuple(args, "sO", &pin_name, &obj))
    {
        if (self->owner)
        {
            // Special cases
            if (PyString_Check(obj))
            {
                FlowString* str_obj = new FlowString(PyString_AsString(obj));
                self->owner->write_pin(pin_name, str_obj);
            }
            else if (PyInt_Check(obj))
            {
                FlowInt* int_obj = new FlowInt(PyInt_AsLong(obj));
                self->owner->write_pin(pin_name, int_obj);
            }
            else if (PyFloat_Check(obj))
            {
                FlowFloat64* flt_obj = new FlowFloat64(PyFloat_AsDouble(obj));
                self->owner->write_pin(pin_name, flt_obj);
            }
            else
            {
                FlowObject* fobj = py_flow_object::owner(obj);
                if (fobj)
                    self->owner->write_pin(pin_name, fobj);
            }
        }

        Py_INCREF(Py_None);
        ret = Py_None;
    }
    return ret;
}


static PyMemberDef PyFlowContext_members[] = {
    { NULL }  /* Sentinel */
};


static PyMethodDef PyFlowContext_methods[] = {
    { "read_pin", (PyCFunction)PyFlowContext_read_pin, METH_VARARGS, "Reads the specified pin." },
    { "write_pin", (PyCFunction)PyFlowContext_write_pin, METH_VARARGS, "Writes object to the specified pin." },
    { NULL }  /* Sentinel */
};

static PyTypeObject PyFlowContextType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "flow.Context",             /*tp_name*/
    sizeof(PyFlowContext),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyFlowContext_dealloc, /*tp_dealloc*/
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
    "FlowContext",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    PyFlowContext_methods,             /* tp_methods */
    PyFlowContext_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyFlowContext_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyFlowContext_new,                 /* tp_new */
};

PyObject* py_flow_module::create_context_type()
{
    if (PyType_Ready(&PyFlowContextType) < 0)
        return nullptr;

    Py_INCREF(&PyFlowContextType);
    return (PyObject *)&PyFlowContextType;
}
PyObject* py_flow_module::create_context(FlowContext* owner)
{
    PyFlowContext* obj = nullptr;
    PyTypeObject* type = &PyFlowContextType;

    PyObject* args = PyTuple_New(0);
    obj = (PyFlowContext*)type->tp_new(type, args, nullptr);
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

