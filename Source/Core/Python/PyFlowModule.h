#ifndef __CORE_PY_FLOW_SYSTEM_H__
#define __CORE_PY_FLOW_SYSTEM_H__

namespace py_flow_module
{
    void init_module();

    void register_object_type(const char* name, PyObject* type);
}

#endif // __CORE_PY_FLOW_SYSTEM_H__
