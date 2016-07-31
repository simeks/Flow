#include "Common.h"

#include "FlowClass.h"
#include "Field.h"


FlowClass::FlowClass(const char* name, uint32_t size, CreateObjectFn creator)
    : FlowType(name, size),
    _creator(creator),
    _super(nullptr)
{
}

FlowClass* FlowClass::super_class() const
{
    return _super;
}
bool FlowClass::is_class(FlowClass* type) const
{
    for (const FlowClass* t = this; t; t = t->super_class())
    {
        if (t == type)
            return true;
    }
    return false;
}
void FlowClass::set_super(FlowClass* super)
{
    _super = super;
}
FlowObject* FlowClass::create_object() const
{
    return _creator();
}
Field* FlowClass::find_field(const std::string& name) const
{
    for (const FlowClass* t = this; t; t = t->super_class())
    {
        for (auto& f : t->_fields)
        {
            if (name == f->name())
                return f.get();
        }
    }
    return nullptr;
}
std::vector<Field*> FlowClass::fields() const
{
    std::vector<Field*> ret;
    for (const FlowClass* t = this; t; t = t->super_class())
    {
        for (auto& f : t->_fields)
        {
            ret.push_back(f.get());
        }
    }
    return ret;
}

