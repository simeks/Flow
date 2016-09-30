#ifndef __FLOW_SCRIPT_OBJECT_H__
#define __FLOW_SCRIPT_OBJECT_H__

#include <Flow/FlowObject.h>

class CORE_API ScriptObject : public FlowObject
{
    DECLARE_SCRIPT_OBJECT(ScriptObject, FlowObject);

public:
    ScriptObject(PyObject* obj = nullptr);
    ~ScriptObject();

};

#endif // __FLOW_SCRIPT_OBJECT_H__
