#ifndef __CORE_PYTHON_FLOW_OBJECT_H__
#define __CORE_PYTHON_FLOW_OBJECT_H__

class FlowObject;
class FlowClass;
namespace py_flow_object
{
    // Type functions
    void type_dealloc(PyObject* self);
    PyObject* type_new(PyTypeObject* type, PyObject*, PyObject*);
    int type_init(PyObject* self);
    size_t type_basicsize();

    // Creates the base object type used as a base type for all implemented objects
    PyObject* create_base_type();
    // Returns the base type (Assumes type is initialized)
    PyTypeObject* base_type();

    // Creates a new python object
    PyObject* create_object(PyTypeObject* type, FlowObject* owner);

    // Returns the owner of the specified object
    FlowObject* owner(PyObject* object);
    void set_owner(PyObject* object, FlowObject* owner);

    struct Type
    {
        const char* name;
        PyObject* type;
		FlowClass* flow_type;
    };
    void register_object_type(const char* name, PyObject* type, FlowClass* flow_type);

    Type* object_types();
    int num_object_types();

    Type* find_object_type(const char* name);
}

#endif // __CORE_PYTHON_FLOW_OBJECT_H__
