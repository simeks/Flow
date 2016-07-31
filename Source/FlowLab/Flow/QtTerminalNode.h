#ifndef __QT_TERMINAL_NODE_H__
#define __QT_TERMINAL_NODE_H__

#include "QtFlowNode.h"

class QLabel;
class QtTerminalNode : public QtBaseNode
{
public:
    enum { Type = UserType + 9 };

    QtTerminalNode(FlowNodePtr node, QGraphicsItem* parent = nullptr);
    ~QtTerminalNode();

    const std::string& var_name() const;
    void set_var_name(const std::string& name);

    std::string value_str() const;
    void set_value(const std::string& value);

    int type() const;

    virtual void save(JsonObject& obj) const OVERRIDE;
    virtual void load(const JsonObject& obj) OVERRIDE;

private:
    QLabel* _name_label;

    void setup();
    void update_label();
};

#endif // __QT_TERMINAL_NODE_H__
