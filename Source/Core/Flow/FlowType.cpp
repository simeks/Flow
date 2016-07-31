#include "Common.h"

#include "FlowType.h"


namespace flow_types
{
    FlowType* g_head = nullptr;

    void register_type(FlowType* type)
    {
        if (g_head)
        {
            type->_list_next = g_head;
        }
        g_head = type;
    }
    void clear_types()
    {
        while (g_head)
        {
            FlowType* type = g_head;
            g_head = g_head->_list_next;
            delete type;
        }
    }
}


FlowType::FlowType(const char* name, uint32_t size)
    : _name(name),
    _size(size),
    _list_next(nullptr)
{
	flow_types::register_type(this);
}

const char* FlowType::name() const
{
    return _name;
}

uint32_t FlowType::size() const
{
    return _size;
}
