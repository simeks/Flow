#ifndef __QT_FLOW_DIAGRAM_SCENE_H__
#define __QT_FLOW_DIAGRAM_SCENE_H__

#include <QGraphicsScene>

#include <Core/Flow/FlowNode.h>
#include <Core/Flow/FlowGraph.h>

class FlowGraph;
class FlowSystem;
class QtBaseNode;
class QtFlowNode;
class QtFlowDiagramScene : public QGraphicsScene
{
    Q_OBJECT

public:
    QtFlowDiagramScene(QObject *parent = nullptr);
    ~QtFlowDiagramScene();

    void create_node(FlowNodePtr node, const QPointF& pos);
    void add_node(QtBaseNode* node);
    void remove_node(QtFlowNode* node);

    void clear_scene();

    FlowGraphPtr flow_graph() const;

    bool save_to_file(const QString& file) const;
    bool load_from_file(const QString& file);

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void drawBackground(QPainter * painter, const QRectF & rect);

private:
    FlowGraphPtr _flow_graph;

    std::map<Guid, QtBaseNode*> _nodes;

signals:
    void scene_cleared();
};


#endif // __QT_FLOW_DIAGRAM_SCENE_H__

