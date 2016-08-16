#include "Common.h"

#include "FlowVector.h"
#include "Python/PythonWrapper.h"


IMPLEMENT_OBJECT(FlowVec3i, "FlowVec3i");
FlowVec3i::FlowVec3i()
{
}
FlowVec3i::FlowVec3i(const Vec3i& v) : _v(v)
{
}
FlowVec3i::~FlowVec3i()
{
}
void FlowVec3i::set(const Vec3i& v)
{
    _v = v;
}
const Vec3i& FlowVec3i::get() const
{
    return _v;
}
PyObject* FlowVec3i::py_create_object()
{
    PyObject* t = PyTuple_New(3);
    PyTuple_SetItem(t, 0, PyLong_FromLong(_v.x));
    PyTuple_SetItem(t, 1, PyLong_FromLong(_v.y));
    PyTuple_SetItem(t, 2, PyLong_FromLong(_v.z));
    return t;
}


IMPLEMENT_OBJECT(FlowVec3d, "FlowVec3d");
FlowVec3d::FlowVec3d()
{
}
FlowVec3d::FlowVec3d(const Vec3d& v) : _v(v)
{
}
FlowVec3d::~FlowVec3d()
{
}
void FlowVec3d::set(const Vec3d& v)
{
    _v = v;
}
const Vec3d& FlowVec3d::get() const
{
    return _v;
}
PyObject* FlowVec3d::py_create_object()
{
    PyObject* t = PyTuple_New(3);
    PyTuple_SetItem(t, 0, PyFloat_FromDouble(_v.x));
    PyTuple_SetItem(t, 1, PyFloat_FromDouble(_v.y));
    PyTuple_SetItem(t, 2, PyFloat_FromDouble(_v.z));
    return t;
}

