#ifndef __CORE_FLOW_OBJECT_H__
#define __CORE_FLOW_OBJECT_H__

#include "FlowClass.h"

typedef void(*PopulateClassFunc)(FlowClass&);

#define DECLARE_OBJECT(TClass, TSuperClass) \
    public: \
    typedef TSuperClass Super; \
    typedef TClass ThisClass; \
    \
    static FlowObject* create_object(); \
    static FlowClass* static_class(); \
    virtual FlowClass* get_class() const; \
    virtual void copy_object_to(FlowObject* object) const;

#define DECLARE_SCRIPT_OBJECT(TClass, TSuperClass) \
    DECLARE_OBJECT(TClass, TSuperClass); \
    public: \
        static PyTypeObject* py_type(); \
        static FlowObject* create_object(PyObject* py_object); \
    private: \
        static bool _py_type_initialized; \
        static int py_##TClass##_init(PyObject* self, PyObject* args, PyObject* kwds); \
        static PyTypeObject py_##TClass##_type; \
        virtual PyObject* py_create_object(); 

#define IMPLEMENT_OBJECT2(TClass, Name, PopulateFn) \
    FlowObject* TClass::create_object() \
    { \
        return new TClass(); \
    } \
    static FlowClass* _##TClass##_type = TClass::static_class(); \
    FlowClass* TClass::static_class() \
    { \
        static FlowClass* type = nullptr; \
        if (!type) \
        { \
            type = new FlowClass(Name, sizeof(TClass), TClass::create_object); \
            if (type != TClass::Super::static_class()) \
            { \
                type->set_super(TClass::Super::static_class()); \
            } \
            PopulateClassFunc fn = PopulateFn; \
            if (fn) \
            { \
                fn(*type); \
            } \
        } \
        return type; \
    } \
    FlowClass* TClass::get_class() const \
    { \
        return TClass::static_class(); \
    } \
    void TClass::copy_object_to(FlowObject* object) const \
    { \
        if (object->get_class() == TClass::static_class()) \
        { \
            *static_cast<TClass*>(object) = *this; \
        } \
    }

#define IMPLEMENT_OBJECT(TClass, Name) IMPLEMENT_OBJECT2(TClass, Name, nullptr)


#define IMPLEMENT_SCRIPT_OBJECT(TClass, Name, ScriptTypeName, MethodTable) \
    IMPLEMENT_OBJECT(TClass, Name); \
    int TClass::py_##TClass##_init(PyObject* self, PyObject* args, PyObject* kwds) \
    { \
        if(!py_flow_object::owner(self)) \
        { \
            py_flow_object::set_owner(self, TClass::create_object(self)); \
            return object_cast<TClass>(py_flow_object::owner(self))->script_object_init(self, args, kwds); \
        } \
        return 0; \
    } \
    PyTypeObject TClass::py_##TClass##_type = { \
        PyObject_HEAD_INIT(NULL) \
        0,                         /*ob_size*/ \
        "flow." ScriptTypeName,             /*tp_name*/ \
        (Py_ssize_t)py_flow_object::type_basicsize(),             /*tp_basicsize*/ \
        0,                         /*tp_itemsize*/ \
        (destructor)py_flow_object::type_dealloc, /*tp_dealloc*/ \
        0,                         /*tp_print*/ \
        0,                         /*tp_getattr*/ \
        0,                         /*tp_setattr*/ \
        0,                         /*tp_compare*/ \
        0,                         /*tp_repr*/ \
        0,                         /*tp_as_number*/ \
        0,                         /*tp_as_sequence*/ \
        0,                         /*tp_as_mapping*/ \
        0,                         /*tp_hash */ \
        0,                         /*tp_call*/ \
        0,                         /*tp_str*/ \
        0,                         /*tp_getattro*/ \
        0,                         /*tp_setattro*/ \
        0,                         /*tp_as_buffer*/ \
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/ \
        Name,           /* tp_doc */ \
        0,		               /* tp_traverse */ \
        0,		               /* tp_clear */ \
        0,		               /* tp_richcompare */ \
        0,		               /* tp_weaklistoffset */ \
        0,		               /* tp_iter */ \
        0,		               /* tp_iternext */ \
        MethodTable,             /* tp_methods */ \
        0,             /* tp_members */ \
        0,                         /* tp_getset */ \
        0,                         /* tp_base */ \
        0,                         /* tp_dict */ \
        0,                         /* tp_descr_get */ \
        0,                         /* tp_descr_set */ \
        0,                         /* tp_dictoffset */ \
        (initproc)py_##TClass##_init,      /* tp_init */ \
        0,                         /* tp_alloc */ \
        py_flow_object::type_new,                 /* tp_new */ \
        }; \
    \
    static PyTypeObject* TClass##_py_type_init = TClass::py_type(); \
    bool TClass::_py_type_initialized = false; \
    PyObject* TClass::py_create_object() \
    { \
        return py_flow_object::create_object(TClass::py_type(), (FlowObject*)this); \
    } \
    PyTypeObject* TClass::py_type() \
    { \
        if (!_py_type_initialized) \
        { \
            py_##TClass##_type.tp_base = py_flow_object::base_type(); \
            py_flow_object::register_object_type(ScriptTypeName, (PyObject*)&py_##TClass##_type, TClass::static_class()); \
            \
            _py_type_initialized = true; \
        } \
        return &py_##TClass##_type; \
    } \
    FlowObject* TClass::create_object(PyObject* py_object) \
    { \
        FlowObject* obj = TClass::create_object(); \
        obj->set_script_object(py_object); \
        return obj; \
    }

class CORE_API FlowObject
{
    DECLARE_SCRIPT_OBJECT(FlowObject, FlowObject);

public:
    FlowObject();
    virtual ~FlowObject();

    FlowObject* clone() const;

    bool is_a(FlowClass* type) const;

    template<typename T>
    bool is_a() const { return is_a(T::static_class()); }

    PyObject* script_object();
    void set_script_object(PyObject* obj);

    template<typename TField>
    void set_field_value(TField* field, const typename TField::FieldType& value);

    template<typename TField>
    const typename TField::FieldType& field_value(TField* field) const;

    void set_field_value(Field* field, const std::string& value);

    std::string field_as_string(Field* field) const;

    FlowObject(const FlowObject& other);
    FlowObject& operator=(const FlowObject& other);

protected:
    virtual void field_updated(Field* /*field*/) { }
    virtual int script_object_init(PyObject* /*self*/, PyObject* /*args*/, PyObject* /*kwds*/) { return 0; }

    PyObject* _py_object;

};

template<typename TField>
void FlowObject::set_field_value(TField* field, const typename TField::FieldType& value)
{
    field->set_value(this, value);
}

template<typename TField>
const typename TField::FieldType& FlowObject::field_value(TField* field) const
{
    return field->value(this);
}

template<typename Type>
Type* object_cast(FlowObject* object)
{
    if (!object || !object->is_a(Type::static_class()))
        return nullptr;

    return (Type*)object;
}


#endif // __CORE_FLOW_OBJECT_H__
