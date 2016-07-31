#ifndef __QT_FLOW_DIAGRAM_VIEW_H__
#define __QT_FLOW_DIAGRAM_VIEW_H__

#include <QGraphicsView>

class Core;
class QMenu;
class QtBaseNode;
class QtFlowConnection;
class QtFlowDiagramScene;
class QtFlowPin;
class QtFlowDiagramView : public QGraphicsView
{
    Q_OBJECT

public:
    QtFlowDiagramView(QWidget *parent = nullptr);
    ~QtFlowDiagramView();

    void set_flow_scene(QtFlowDiagramScene* scene);
    QtFlowDiagramScene* flow_scene() const;

protected:
    void mousePressEvent(QMouseEvent* mouse_event);
    void mouseMoveEvent(QMouseEvent* mouse_event);
    void mouseReleaseEvent(QMouseEvent* mouse_event);
    void keyPressEvent(QKeyEvent *e);
    void wheelEvent(QWheelEvent *e);

private:
    enum Mode
    {
        Mode_Nothing,
        Mode_Move,
        Mode_DragPin,
        Mode_Scroll
    };

    QMenu* _context_menu;

    QtFlowDiagramScene* _current_scene;

    QPoint _last_mouse_pos;

    Mode _mode;

    QtFlowPin* _selected_pin;
    QtFlowPin* _temp_pin;
    QtFlowConnection* _temp_connection;
    QtFlowPin* _highlight_pin;

    void build_node_menu();

private slots:
    void show_context_menu(const QPoint&);
    void scene_cleared();

signals:
    void flow_node_selected(QtBaseNode* node);

};


#endif // __QT_FLOW_DIAGRAM_VIEW_H__

