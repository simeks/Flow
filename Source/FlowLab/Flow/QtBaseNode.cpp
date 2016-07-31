#include <Core/Common.h>

#include <Core/Flow/FlowSystem.h>
#include <Core/Json/JsonObject.h>

#include "QtBaseNode.h"
#include "QtFlowConnection.h"
#include "QtFlowPin.h"

#include <QGraphicsGridLayout>
#include <QPainter>

QtBaseNode::QtBaseNode(FlowNodePtr node, QGraphicsItem* parent) : QGraphicsWidget(parent), _node(node)
{
    _layout = new QGraphicsGridLayout(this);
    _layout->setContentsMargins(2, 2, 2, 2);

    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsFocusable, true);
}
QtBaseNode::~QtBaseNode()
{
}


void QtBaseNode::clear_pins()
{
    for (auto& pin : _pins)
    {
        while (pin->connection_count())
        {
            QtFlowConnection* conn = pin->connection(0);
            conn->unlink();
        }

        delete pin;
    }
    _pins.clear();
}
QRectF QtBaseNode::boundingRect() const
{
    return contentsRect();
}
void QtBaseNode::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setRenderHint(QPainter::Antialiasing);

    painter->setBrush(QBrush(Qt::lightGray));

    if (isSelected())
        painter->setPen(QPen(Qt::black, 2, Qt::DashLine));

    painter->drawRect(contentsRect());
}

FlowNodePtr QtBaseNode::node() const
{
    return _node;
}
Guid QtBaseNode::node_id() const
{
    if (_node)
        return _node->node_id();
    return Guid();
}
const std::vector<QtFlowPin*>& QtBaseNode::pins() const
{
    return _pins;
}
void QtBaseNode::save(JsonObject& obj) const
{
    obj["id"].set_string(guid::to_string(node_id()));
    obj["scene_pos"].set_empty_array();
    obj["scene_pos"].append().set_double(pos().x());
    obj["scene_pos"].append().set_double(pos().y());

    obj["node_class"].set_string(_node->node_class());
}
void QtBaseNode::load(const JsonObject& obj)
{
    std::string node_class = obj["node_class"].as_string();

    FlowNode* tpl = FlowSystem::get().node_template(node_class);
    _node.reset((FlowNode*)tpl->clone());
    _node->set_node_id(guid::from_string(obj["id"].as_string()));

    const JsonObject& json_pos = obj["scene_pos"];
    setPos(QPointF(json_pos[0].as_double(), json_pos[1].as_double()));
}
