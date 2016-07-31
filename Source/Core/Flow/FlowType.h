#ifndef __CORE_FLOW_TYPE_H__
#define __CORE_FLOW_TYPE_H__

#include <Core/Common.h>

class FlowObject;
class FlowType;

namespace flow_types
{
    CORE_API void register_type(FlowType* type);

    /// Clears all registered types. 
    /// Should only be called during application shut down.
    void clear_types();
}

class CORE_API FlowType
{
public:
    FlowType(const char* name, uint32_t size);

    const char* name() const;
    uint32_t size() const;

    friend void flow_types::register_type(FlowType* type);
    friend void flow_types::clear_types();
private:
    const char* _name;
    uint32_t _size;

    FlowType* _list_next;
};

#endif // __CORE_FLOW_TYPE_H__
