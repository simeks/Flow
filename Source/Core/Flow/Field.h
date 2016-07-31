#ifndef __FLOW_FIELD_H__
#define __FLOW_FIELD_H__

#include "FlowObject.h"

class CORE_API Field : public FlowObject
{
    DECLARE_OBJECT(Field, FlowObject);
public:
    Field();
    Field(const char* name, size_t offset, uint32_t flags);

    const char* name() const;
    uint32_t flags() const;

protected:
    const char* _name;
    size_t _offset;
    uint32_t _flags;
};

template<typename T>
class TField : public Field
{
public:
    typedef T FieldType;

    TField();
    TField(const char* name, size_t offset, uint32_t flags);

    T* value_ptr(const FlowObject* object) const;
    const T& value(const FlowObject* object) const;
    void set_value(const FlowObject* object, const T& value) const;

};

#define DECLARE_FIELD_TYPE(Name, CppType) \
    class CORE_API Name : public TField<CppType> \
    { \
        DECLARE_OBJECT(Name, Field); \
    public: \
        Name(); \
        Name(const char* name, size_t offset, uint32_t flags); \
    };

#define IMPLEMENT_FIELD_TYPE(Name, CppType) \
    IMPLEMENT_OBJECT(Name, #Name) \
    Name::Name() \
    { \
    } \
    Name::Name(const char* name, size_t offset, uint32_t flags) : TField(name, offset, flags) \
    { \
    }


DECLARE_FIELD_TYPE(Int32Field, int32_t);
DECLARE_FIELD_TYPE(UInt32Field, uint32_t);
DECLARE_FIELD_TYPE(Int64Field, int64_t);
DECLARE_FIELD_TYPE(UInt64Field, uint64_t);
DECLARE_FIELD_TYPE(Float32Field, float);
DECLARE_FIELD_TYPE(Float64Field, double);
DECLARE_FIELD_TYPE(StringField, std::string);

#include "Field.inl"

#endif // __FLOW_FIELD_H__
