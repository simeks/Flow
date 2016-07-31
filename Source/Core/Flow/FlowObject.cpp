#include "Common.h"

#include "FlowObject.h"
#include "Field.h"

#include "Python/PyFlowModule.h"
#include "Python/PyFlowObject.h"
#include "Python/PythonWrapper.h"

static PyMethodDef py_FlowObject_methods[] = {
    { NULL }  /* Sentinel */
};

IMPLEMENT_SCRIPT_OBJECT(FlowObject, "FlowObject", "Object", py_FlowObject_methods);

FlowObject::FlowObject() : _py_object(nullptr)
{
}
FlowObject::~FlowObject() 
{
    Py_XDECREF(_py_object);
}
FlowObject* FlowObject::clone() const
{
    FlowObject* clone = get_class()->create_object();
    copy_object_to(clone);
    return clone;
}
bool FlowObject::is_a(FlowClass* type) const
{
    if (type == nullptr)
        return true;

    for (FlowClass* t = this->get_class(); t; t = t->super_class())
    {
        if (t == type)
            return true;
    }
    return false;
}

PyObject* FlowObject::script_object()
{
    if (!_py_object)
        _py_object = py_create_object();
    return _py_object;
}
void FlowObject::set_script_object(PyObject* obj)
{
    _py_object = obj;
    Py_XINCREF(_py_object);
}

void FlowObject::set_field_value(Field* field, const std::string& value)
{
    if (field->is_a(StringField::static_class()))
    {
        set_field_value((StringField*)field, value);
    }
    else if (field->get_class() == Int32Field::static_class())
    {
        set_field_value((Int32Field*)field, strtol(value.c_str(), 0, 0));
    }
    else if (field->get_class() == UInt32Field::static_class())
    {
        set_field_value((UInt32Field*)field, strtoul(value.c_str(), 0, 0));
    }
    else if (field->get_class() == Int64Field::static_class())
    {
        set_field_value((Int64Field*)field, strtoll(value.c_str(), 0, 0));
    }
    else if (field->get_class() == UInt64Field::static_class())
    {
        set_field_value((UInt64Field*)field, strtoull(value.c_str(), 0, 0));
    }
    else if (field->get_class() == Float32Field::static_class())
    {
        set_field_value((Float32Field*)field, strtof(value.c_str(), 0));
    }
    else if (field->get_class() == Float64Field::static_class())
    {
        set_field_value((Float64Field*)field, strtod(value.c_str(), 0));
    }
    else
    {
        FATAL_ERROR("Could not convert from string to field type.");
    }
}
std::string FlowObject::field_as_string(Field* field) const
{
    std::stringstream s;
    if (field->is_a(StringField::static_class()))
    {
        return field_value((StringField*)field);
    }
    else if (field->get_class() == Int32Field::static_class())
    {
        s << field_value((Int32Field*)field);
    }
    else if (field->get_class() == UInt32Field::static_class())
    {
        s << field_value((UInt32Field*)field);
    }
    else if (field->get_class() == Int64Field::static_class())
    {
        s << field_value((Int64Field*)field);
    }
    else if (field->get_class() == UInt64Field::static_class())
    {
        s << field_value((UInt64Field*)field);
    }
    else if (field->get_class() == Float32Field::static_class())
    {
        s << field_value((Float32Field*)field);
    }
    else if (field->get_class() == Float64Field::static_class())
    {
        s << field_value((Float64Field*)field);
    }
    else
    {
        FATAL_ERROR("Could not convert from string to field type.");
    }
    return s.str();
}

FlowObject::FlowObject(const FlowObject& )
{
    _py_object = nullptr;
}
FlowObject& FlowObject::operator=(const FlowObject& )
{
    _py_object = nullptr;
    return *this;
}

