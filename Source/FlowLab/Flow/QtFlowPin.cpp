#include <Core/Common.h>

#include "QtFlowConnection.h"
#include "QtFlowNode.h"
#include "QtFlowPin.h"

#include <QLabel>
#include <QPainter>

QtFlowPin::QtFlowPin(FlowPinPtr pin, QtBaseNode* owner)
    : QGraphicsProxyWidget(owner),
    _pin(pin),
    _owner(owner)
{
    QLabel* label = new QLabel(">");
    if (_pin->pin_type() == FlowPin::In)
        label->setContentsMargins(5, 0, 0, 0);
    else
        label->setContentsMargins(0, 0, 5, 0);

    label->setAttribute(Qt::WA_TranslucentBackground);
    setWidget(label);
}
QtFlowPin::~QtFlowPin()
{
}

void QtFlowPin::set_highlighted(bool highlight)
{
    QLabel* label = (QLabel*)widget();
    if (highlight)
    {
        label->setText("O");
    }
    else
    {
        label->setText(">");
    }
}
bool QtFlowPin::add_connection(QtFlowConnection* connection)
{
    // Only one connection per input pin
    if (pin_type() == FlowPin::In && !_connections.empty())
        return false;

    // Make sure pins match
    QtFlowPin* other = (pin_type() == FlowPin::In) ? connection->start_pin() : connection->end_pin();
    if (pin_type() == other->pin_type())
        return false;

    // Make sure pins are not already connected
    for (auto& c : _connections)
    {
        if (c->start_pin() == other || c->end_pin() == other)
            return false;
    }

    // Check that data types match
    FlowPinPtr out_pin = connection->start_pin()->pin();
    FlowPinPtr in_pin = connection->end_pin()->pin();
    if (out_pin->pin_type() == in_pin->pin_type())
        return false;

    connection->set_pin(this);
    _connections.push_back(connection);

    _pin->link_to(other->pin().get());

    return true;

}
void QtFlowPin::remove_connection(QtFlowConnection* connection)
{
    auto it = std::find(_connections.begin(), _connections.end(), connection);
    if (it != _connections.end())
        _connections.erase(it);
}
const std::vector<QtFlowConnection*>& QtFlowPin::connections() const
{
    return _connections;
}
QtFlowConnection* QtFlowPin::connection(int i) const
{
    return _connections[i];
}
size_t QtFlowPin::connection_count() const
{
    return _connections.size();
}
FlowPinPtr QtFlowPin::pin() const
{
    return _pin;
}
FlowPin::Type QtFlowPin::pin_type() const
{
    return _pin->pin_type();
}
int QtFlowPin::pin_id() const
{
    return _pin->pin_id();
}
QtBaseNode* QtFlowPin::owner() const
{
    return _owner;
}
int QtFlowPin::type() const
{
    return Type;
}
