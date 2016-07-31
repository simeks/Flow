#ifndef __QT_FLOW_PIN_H__
#define __QT_FLOW_PIN_H__

#include <QGraphicsProxyWidget>

#include <Core/Flow/FlowPin.h>

class QtFlowConnection;
class QtBaseNode;
class QtFlowPin : public QGraphicsProxyWidget
{
    Q_OBJECT

public:
    enum { Type = UserType + 3 };

    QtFlowPin(FlowPinPtr pin, QtBaseNode* owner);
    ~QtFlowPin();

    void set_highlighted(bool highlight);
    bool add_connection(QtFlowConnection* connection);
    void remove_connection(QtFlowConnection* connection);

    const std::vector<QtFlowConnection*>& connections() const;
    QtFlowConnection* connection(int i) const;
    size_t connection_count() const;

    FlowPinPtr pin() const;
    FlowPin::Type pin_type() const;
    int pin_id() const;

    QtBaseNode* owner() const;

    int type() const;

private:
    FlowPinPtr _pin;
    std::vector<QtFlowConnection*> _connections;

    QtBaseNode* _owner;
};

#endif // __QT_FLOW_PIN_H__