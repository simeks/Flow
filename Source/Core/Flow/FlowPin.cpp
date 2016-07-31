#include "Common.h"

#include "FlowPin.h"

IMPLEMENT_OBJECT(FlowPin, "FlowPin");
FlowPin::FlowPin()
: _pin_type(Unknown),
  _owner(nullptr),
  _id(-1)
{

}
FlowPin::FlowPin(const std::string& name,
                 Type pin_type, 
                 FlowNode* owner, 
                 int id)
: _pin_type(pin_type),
  _name(name),
  _owner(owner),
  _id(id)
{
}
FlowPin::~FlowPin()
{
    unlink_all();
}
void FlowPin::link_to(FlowPin* pin)
{
    if (pin == this)
        return;

    auto it = std::find(_links.begin(), _links.end(), pin);
    if (it == _links.end())
    {
        _links.push_back(pin);
    }
}
void FlowPin::unlink(FlowPin* pin)
{
    auto it = std::find(_links.begin(), _links.end(), pin);
    if (it != _links.end())
    {
        _links.erase(it);
    }
}
void FlowPin::unlink_all()
{
    for (auto& l : _links)
    {
        l->unlink(this);
    }
    _links.clear();
}
const std::vector<FlowPin*> FlowPin::links() const
{
    return _links;
}
bool FlowPin::is_linked() const
{
    return !_links.empty();
}
FlowPin::Type FlowPin::pin_type() const
{
    return _pin_type;
}
const std::string& FlowPin::name() const
{
    return _name;
}
FlowNode* FlowPin::owner() const
{
    return _owner;
}
int FlowPin::pin_id() const
{
    return _id;
}


