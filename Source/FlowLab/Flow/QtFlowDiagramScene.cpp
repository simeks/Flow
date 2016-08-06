#include <Core/Common.h>
#include <Core/Flow/FlowGraph.h>
#include <Core/Flow/FlowSystem.h>
#include <Core/Flow/TerminalNode.h>
#include <Core/Json/Json.h>
#include <Core/Json/JsonObject.h>

#include "QtFlowConnection.h"
#include "QtFlowDiagramScene.h"
#include "QtFlowPin.h"
#include "QtFlowNode.h"
#include "QtTerminalNode.h"

#include <QtGui>
#include <QGraphicsSceneMouseEvent>

QtFlowDiagramScene::QtFlowDiagramScene(QObject *parent)
    : QGraphicsScene(parent)
{
    _flow_graph = FlowSystem::get().create_graph();
}

QtFlowDiagramScene::~QtFlowDiagramScene()
{
    clear_scene();
}

void QtFlowDiagramScene::create_node(FlowNodePtr node, const QPointF& pos)
{
    QtBaseNode* ui_node = nullptr;
    if (node->is_a(TerminalNode::static_class()))
    {
        ui_node = new QtTerminalNode(node);
    }
    else
    {
        ui_node = new QtFlowNode(node);
    }
    ui_node->setPos(pos);
    addItem(ui_node);

    _flow_graph->add_node(node);
    _nodes[node->node_id()] = ui_node;
}
void QtFlowDiagramScene::add_node(QtBaseNode* node)
{
    addItem(node);
    _flow_graph->add_node(node->node());
    _nodes[node->node_id()] = node;
}
void QtFlowDiagramScene::remove_node(QtFlowNode* node)
{
    auto it = _nodes.find(node->node_id());
    if (it != _nodes.end())
        _nodes.erase(it);

    _flow_graph->remove_node(node->node());

    node->clear_pins();
    removeItem(node);
    delete node;
}
void QtFlowDiagramScene::clear_scene()
{
    for (auto& node : _nodes)
    {
        node.second->clear_pins();
        removeItem(node.second);
        delete node.second;
    }
    _nodes.clear();

    clear();
    _flow_graph->clear();

    emit scene_cleared();
}

void QtFlowDiagramScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    QGraphicsScene::mouseMoveEvent(mouseEvent);
}
void QtFlowDiagramScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    const QRectF nrect = rect.normalized();
    painter->save();
    painter->setPen(QPen(Qt::lightGray, 1));
    int l = int(nrect.left());
    l -= (l % 10);

    int r = int(nrect.right());
    r -= (r % 10);
    if (r < int(nrect.right()))
        r += 10;

    int t = int(nrect.top());
    t -= (t % 10);

    int b = int(nrect.bottom());
    b -= (b % 10);
    if (b < int(nrect.bottom()))
        b += 10;

    for (int x = l; x <= r; x += 10)
        for (int y = t; y <= b; y += 10)
            painter->drawPoint(x, y);

    painter->restore();
}
FlowGraphPtr QtFlowDiagramScene::flow_graph() const
{
    return _flow_graph;
}
bool QtFlowDiagramScene::save_to_file(const QString& file) const
{
    JsonObject root;
    root.set_empty_object();

    QFile f(file);
    if (!f.open(QIODevice::WriteOnly))
        return false;

    JsonObject& nodes = root["nodes"];
    nodes.set_empty_array();

    for (auto& node : _nodes)
    {
        JsonObject& node_obj = nodes.append();
        node_obj.set_empty_object();

        node.second->save(node_obj);
    }

    JsonObject& links = root["links"];
    links.set_empty_array();

    for (auto& node : _nodes)
    {
        QtBaseNode* base_node = node.second;
        for (auto& out_pin : base_node->pins())
        {
            if (out_pin->pin_type() == FlowPin::Out)
            {
                for (auto& conn : out_pin->connections())
                {
                    QtFlowPin* in_pin = conn->end_pin();

                    JsonObject& link = links.append();
                    link.set_empty_object();

                    link["out_node"].set_string(guid::to_string(out_pin->owner()->node_id()));
                    link["out_pin"].set_int(out_pin->pin_id());

                    link["in_node"].set_string(guid::to_string(in_pin->owner()->node_id()));
                    link["in_pin"].set_int(in_pin->pin_id());
                }
            }
        }
    }

    JsonWriter writer;
    return writer.write_file(root, file.toStdString(), true);
}
bool QtFlowDiagramScene::load_from_file(const QString& file)
{
    clear_scene();

    JsonObject root;

    JsonReader reader;
    if (!reader.read_file(file.toStdString(), root))
    {
        console::error("Failed to read graph file: %s.", reader.error_message().c_str());
        return false;
    }

    const JsonObject& nodes = root["nodes"];
    for (auto node_obj : nodes.as_array())
    {
        QtBaseNode* node = nullptr;
        if (node_obj["node_class"].as_string().compare(0, 12, "TerminalNode") == 0)
        {
            node = new QtTerminalNode(nullptr);
        }
        else
        {
            node = new QtFlowNode(nullptr);
        }
        node->load(node_obj);
        add_node(node);
    }

    const JsonObject& links = root["links"];
    for (auto link : links.as_array())
    {
        Guid out_id = guid::from_string(link["out_node"].as_string());
        Guid in_id = guid::from_string(link["in_node"].as_string());

        QtBaseNode* out_node = _nodes.at(out_id);
        QtBaseNode* in_node = _nodes.at(in_id);

        QtFlowPin* out_pin = out_node->pins().at(link["out_pin"].as_int());
        QtFlowPin* in_pin = in_node->pins().at(link["in_pin"].as_int());

        QtFlowConnection* conn_item = new QtFlowConnection(out_pin, in_pin);
        out_pin->add_connection(conn_item);
        in_pin->add_connection(conn_item);
        addItem(conn_item);
    }

    return true;
}
