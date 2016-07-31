#ifndef __CORE_FLOW_CLASS_H__
#define __CORE_FLOW_CLASS_H__

#include "FlowType.h"

class Field;
class CORE_API FlowClass : public FlowType
{
public:
    typedef FlowObject* (*CreateObjectFn)();

    FlowClass(const char* name, uint32_t size, CreateObjectFn creator);

    FlowObject* create_object() const;

    FlowClass* super_class() const;
    void set_super(FlowClass* super);

    bool is_class(FlowClass* c) const;

    Field* find_field(const std::string& name) const;
    std::vector<Field*> fields() const;

    template<typename FieldType>
    void add_field(const char* name, size_t offset, uint32_t flags = 0);

private:
    CreateObjectFn _creator;
    FlowClass* _super; // Super class

    std::vector<std::shared_ptr<Field>> _fields;
};


template<typename FieldType>
void FlowClass::add_field(const char* name, size_t offset, uint32_t flags)
{
    _fields.push_back(std::make_shared<FieldType>(name, offset, flags));
}

#endif // __CORE_FLOW_CLASS_H__
