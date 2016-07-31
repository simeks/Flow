#ifndef __QT_FLOW_NODE_H__
#define __QT_FLOW_NODE_H__

#include "QtBaseNode.h"

class QtFlowNode : public QtBaseNode
{
public:
    enum { Type = UserType + 8 };

    QtFlowNode(FlowNodePtr node, QGraphicsItem* parent = nullptr);
    virtual ~QtFlowNode();

    int type() const;

    virtual void save(JsonObject& obj) const OVERRIDE;
    virtual void load(const JsonObject& ob) OVERRIDE;

private:
    void setup();

};

#endif // __QT_FLOW_NODE_H__