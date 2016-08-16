#ifndef __CORE_FLOW_VECTOR_H__
#define __CORE_FLOW_VECTOR_H__

#include "FlowObject.h"

#include "Image/Vec3.h"

class FlowVec3i : public FlowObject
{
    DECLARE_OBJECT(FlowVec3i, FlowObject);

public:
    FlowVec3i();
    FlowVec3i(const Vec3i& v);
    virtual ~FlowVec3i();

    void set(const Vec3i& v);
    const Vec3i& get() const;

private:
    Vec3i _v;
private:
    virtual PyObject* py_create_object();
};

class FlowVec3d : public FlowObject
{
    DECLARE_OBJECT(FlowVec3d, FlowObject);

public:
    FlowVec3d();
    FlowVec3d(const Vec3d& v);
    virtual ~FlowVec3d();

    void set(const Vec3d& v);
    const Vec3d& get() const;

private:
    Vec3d _v;
private:
    virtual PyObject* py_create_object();
};

#endif // __CORE_FLOW_VECTOR_H__
