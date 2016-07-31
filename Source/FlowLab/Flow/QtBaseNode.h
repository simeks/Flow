#ifndef __QT_BASE_NODE_H__
#define __QT_BASE_NODE_H__

#include <QGraphicsWidget>
#include <Core/Flow/FlowNode.h>

class JsonObject;

class FlowSystem;
class QGraphicsGridLayout;
class QWidget;
class QtFlowPin;
class QtBaseNode : public QGraphicsWidget
{
public:
    QtBaseNode(FlowNodePtr node, QGraphicsItem* parent = nullptr);
    virtual ~QtBaseNode();

    void clear_pins();

    QRectF boundingRect() const;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);

    FlowNodePtr node() const;
    Guid node_id() const;

    const std::vector<QtFlowPin*>& pins() const;

    virtual void save(JsonObject& obj) const;
    virtual void load(const JsonObject& obj);

protected:
    void setup();

    QGraphicsGridLayout* _layout;
    bool _highlighted;

    std::vector<QtFlowPin*> _pins;

    FlowNodePtr _node;
};


#endif // __QT_BASE_NODE_H__