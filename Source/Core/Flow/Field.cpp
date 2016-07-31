#include "Common.h"

#include "Field.h"

IMPLEMENT_OBJECT(Field, "Field");

Field::Field() : _name(""), _offset(0), _flags(0) {}
Field::Field(const char* name, size_t offset, uint32_t flags) : _name(name), _offset(offset), _flags(flags)
{
}
const char* Field::name() const
{
    return _name;
}
uint32_t Field::flags() const
{
    return _flags;
}

IMPLEMENT_FIELD_TYPE(Int32Field, int32_t);
IMPLEMENT_FIELD_TYPE(UInt32Field, uint32_t);
IMPLEMENT_FIELD_TYPE(Int64Field, int64_t);
IMPLEMENT_FIELD_TYPE(UInt64Field, uint64_t);
IMPLEMENT_FIELD_TYPE(Float32Field, float);
IMPLEMENT_FIELD_TYPE(Float64Field, double);
IMPLEMENT_FIELD_TYPE(StringField, std::string);
