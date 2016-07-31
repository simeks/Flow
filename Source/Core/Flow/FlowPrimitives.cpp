#include "Common.h"

#include "FlowPrimitives.h"
#include "Python/PythonWrapper.h"

IMPLEMENT_OBJECT(NumericObject, "NumericObject");
NumericObject::NumericObject()
{
}
NumericObject::~NumericObject()
{
}
PyObject* NumericObject::py_create_object()
{
    if (is_integer())
    {
        return PyInt_FromSize_t(as_int());
    }
    else
    {
        return PyFloat_FromDouble(as_float());
    }
}

IMPLEMENT_NUMERIC_TYPE(FlowInt, int32_t, "Int");
IMPLEMENT_NUMERIC_TYPE(FlowUInt32, uint32_t, "UInt");
IMPLEMENT_NUMERIC_TYPE(FlowInt64, int64_t, "Int64");
IMPLEMENT_NUMERIC_TYPE(FlowUInt64, uint64_t, "UInt64");
IMPLEMENT_NUMERIC_TYPE(FlowFloat, float, "Float");
IMPLEMENT_NUMERIC_TYPE(FlowFloat64, double, "Float64");
