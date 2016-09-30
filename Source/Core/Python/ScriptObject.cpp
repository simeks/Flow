#include "Common.h"

#include "ScriptObject.h"

#include "Python/PyFlowModule.h"
#include "Python/PyFlowObject.h"
#include "Python/PythonWrapper.h"

static PyMethodDef py_ScriptObject_methods[] = {
    { NULL }  /* Sentinel */
};

IMPLEMENT_SCRIPT_OBJECT(ScriptObject, "ScriptObject", "Object", py_ScriptObject_methods);

ScriptObject::ScriptObject(PyObject* obj)
{
    _py_object = obj;
}
ScriptObject::~ScriptObject()
{
}






