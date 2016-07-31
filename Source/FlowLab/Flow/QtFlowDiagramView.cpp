#include <Core/Common.h>
#include <Core/Core.h>
#include <Core/Flow/FlowNode.h>
#include <Core/Flow/FlowSystem.h>

#include "QtFlowConnection.h"
#include "QtFlowDiagramView.h"
#include "QtFlowDiagramScene.h"
#include "QtFlowNode.h"
#include "QtFlowPin.h"
#include "QtTerminalNode.h"

#include <QGraphicsItem>
#include <QMenu>
#include <QMetaType>
#include <QMouseEvent>

Q_DECLARE_METATYPE(FlowNode*);

QtFlowDiagramView::QtFlowDiagramView(QWidget *parent)
    : QGraphicsView(parent),
    _current_scene(nullptr),
    _selected_pin(nullptr),
    _temp_pin(nullptr),
    _temp_connection(nullptr),
    _highlight_pin(nullptr),
    _mode(Mode_Nothing)
{
    build_node_menu();

    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_context_menu(const QPoint&)));

    setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);

    QtFlowDiagramScene* diagram_scene = new QtFlowDiagramScene(this);
    diagram_scene->setSceneRect(0, 0, 5000, 5000);
    set_flow_scene(diagram_scene);
    connect(diagram_scene, SIGNAL(scene_cleared()), this, SLOT(scene_cleared()));

    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

QtFlowDiagramView::~QtFlowDiagramView()
{
}

void QtFlowDiagramView::set_flow_scene(QtFlowDiagramScene* scene)
{
    _current_scene = scene;
    setScene(scene);
}
QtFlowDiagramScene* QtFlowDiagramView::flow_scene() const
{
    return _current_scene;
}
void QtFlowDiagramView::mousePressEvent(QMouseEvent* mouse_event)
{
    if (_current_scene && mouse_event->button() & Qt::MouseButton::LeftButton)
    {
        emit flow_node_selected(nullptr);

        auto scene_items = items(mouse_event->pos());
        if (scene_items.size() != 0)
        {
            for (auto& selected : scene_items)
            {
                if (selected->type() == QtFlowNode::Type || selected->type() == QtTerminalNode::Type)
                {
                    _current_scene->clearSelection();
                    _current_scene->clearFocus();

                    selected->setSelected(true);

                    emit flow_node_selected((QtBaseNode*)selected);

                    _mode = Mode_Move;

                    break;
                }
                else if (selected->type() == QtFlowPin::Type)
                {
                    _current_scene->clearSelection();
                    _current_scene->clearFocus();

                    _selected_pin = (QtFlowPin*)selected;
                    if (_temp_connection)
                        delete _temp_connection;

                    _temp_connection = nullptr;
                    if (_selected_pin->pin_type() == FlowPin::Out)
                    {
                        _temp_pin = new QtFlowPin(std::make_shared<FlowPin>("", FlowPin::In, nullptr, -1), nullptr);
                        _temp_connection = new QtFlowConnection(_selected_pin, _temp_pin);
                    }
                    else
                    {
                        _temp_pin = new QtFlowPin(std::make_shared<FlowPin>("", FlowPin::Out, nullptr, -1), nullptr);
                        _temp_connection = new QtFlowConnection(_temp_pin, _selected_pin);
                    }
                    _temp_pin->setPos(mapToScene(mouse_event->pos()));

                    _current_scene->addItem(_temp_connection);

                    _mode = Mode_DragPin;

                    break;
                }
                else if (selected->type() == QtFlowConnection::Type)
                {
                    _current_scene->clearSelection();
                    _current_scene->clearFocus();
                    selected->setSelected(true);
                    break;
                }

            }
        }
        else
        {
            _mode = Mode_Scroll;
            setDragMode(DragMode::ScrollHandDrag);
            QGraphicsView::mousePressEvent(mouse_event);

            _current_scene->clearSelection();
            _current_scene->clearFocus();
        }
    }
    else
    {
        QGraphicsView::mousePressEvent(mouse_event);
    }
    _last_mouse_pos = mouse_event->pos();
}
void QtFlowDiagramView::mouseMoveEvent(QMouseEvent* mouse_event)
{
    switch (_mode)
    {
    case Mode_Move:
        for (auto& item : _current_scene->selectedItems())
        {
            if (item->type() == QtFlowNode::Type || item->type() == QtTerminalNode::Type)
            {
                QPointF delta = mapToScene(mouse_event->pos()) - mapToScene(_last_mouse_pos);
                item->setPos(item->pos() + delta);
            }
        }
        break;
    case Mode_DragPin:
    {
        if (_highlight_pin)
        {
            _highlight_pin->set_highlighted(false);
            _highlight_pin = nullptr;
        }

        auto scene_items = items(mouse_event->pos());
        for (auto& selected : scene_items)
        {
            if (selected->type() == QtFlowPin::Type)
            {
                QtFlowPin* pin = (QtFlowPin*)selected;

                if (pin != _selected_pin &&
                    pin->pin_type() != _selected_pin->pin_type())
                {
                    // Check if pins match
                    FlowPinPtr in_pin = (_selected_pin->pin_type() == FlowPin::In) ? _selected_pin->pin() : pin->pin();
                    FlowPinPtr out_pin = (_selected_pin->pin_type() == FlowPin::Out) ? _selected_pin->pin() : pin->pin();

                    // Input pin determines supported types
                    //if (out_pin->type()->is_type(in_pin->type()))
                    //{
                    _highlight_pin = pin;
                    _highlight_pin->set_highlighted(true);
                    //}
                }
            }
        }

        _temp_pin->setPos(mapToScene(mouse_event->pos()));
        _temp_pin->update();
        _temp_connection->update();

        break;
    }
    case Mode_Scroll:
    {
        QGraphicsView::mouseMoveEvent(mouse_event);
        break;
    }
    default:
        QGraphicsView::mouseMoveEvent(mouse_event);
    };
    _last_mouse_pos = mouse_event->pos();
}
void QtFlowDiagramView::mouseReleaseEvent(QMouseEvent* mouse_event)
{
    switch (_mode)
    {
    case Mode_DragPin:
        if (_temp_pin && _highlight_pin && _selected_pin)
        {
            _temp_connection->set_pin(_highlight_pin);
            _temp_connection->set_pin(_selected_pin);

            if (_highlight_pin->add_connection(_temp_connection) &&
                _selected_pin->add_connection(_temp_connection))
            {
                _highlight_pin->set_highlighted(false);
                _highlight_pin = nullptr;
                _temp_connection = nullptr;
            }
        }

        if (_temp_connection)
        {
            delete _temp_connection;
            _temp_connection = nullptr;
        }

        if (_temp_pin)
        {
            delete _temp_pin;
            _temp_pin = nullptr;
        }
        _selected_pin = nullptr;

        break;
    case Mode_Scroll:
    {
        setDragMode(DragMode::NoDrag);
        QGraphicsView::mouseMoveEvent(mouse_event);
        break;
    }
    default:
        QGraphicsView::mouseMoveEvent(mouse_event);
    };
    _last_mouse_pos = mouse_event->pos();

    _mode = Mode_Nothing;
}
void QtFlowDiagramView::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Delete)
    {
        for (auto& item : _current_scene->selectedItems())
        {
            if (item->type() == QtFlowNode::Type || item->type() == QtTerminalNode::Type)
            {
                QtFlowNode* node = (QtFlowNode*)item;
                _current_scene->remove_node(node);
            }
            else if (item->type() == QtFlowConnection::Type)
            {
                QtFlowConnection* conn = (QtFlowConnection*)item;
                conn->start_pin()->remove_connection(conn);
                conn->end_pin()->remove_connection(conn);
                delete conn;
            }
        }
        _current_scene->clearSelection();
    }

    QGraphicsView::keyPressEvent(e);
}
void QtFlowDiagramView::wheelEvent(QWheelEvent *e)
{
    qreal zoom = e->delta()*0.0005;
    
    QTransform t = transform();
    t.scale(1 + zoom, 1 + zoom);
    setTransform(t);
}

void QtFlowDiagramView::build_node_menu()
{
    _context_menu = new QMenu(this);
    std::map<std::string, QMenu*> sub_menus;
    sub_menus[""] = _context_menu;

    for (auto& node : FlowSystem::get().node_templates())
    {
        std::vector<std::string> elems;

        std::stringstream ss(node->category());
        std::string item;
        while (std::getline(ss, item, '/')) 
        {
            elems.push_back(item);
        }

        QAction* node_action = new QAction(QString::fromStdString(node->title()), _context_menu);
        node_action->setData(QVariant::fromValue<FlowNode*>(node));

        QMenu* target_menu = _context_menu;
        std::string menu_str = "";
        for (int i = 0; i < elems.size(); ++i)
        {
            menu_str += elems[i];
            auto it = sub_menus.find(menu_str);
            if (it == sub_menus.end())
            {
                sub_menus[menu_str] = new QMenu(QString::fromStdString(elems[i]), target_menu);
                target_menu->addMenu(sub_menus[menu_str]);
            }
            target_menu = sub_menus[menu_str];
        }

        target_menu->addAction(node_action);

    }
}

void QtFlowDiagramView::show_context_menu(const QPoint& pt)
{
    if (!_current_scene)
        return;

    auto scene_items = items(pt);
    if (scene_items.size())
        return;

    QPoint global_pt = mapToGlobal(pt);

    QAction* action = _context_menu->exec(global_pt);
    if (action)
    {
        FlowNode* template_node = action->data().value<FlowNode*>();
        if (template_node)
        {
            FlowNodePtr node((FlowNode*)template_node->clone());
            node->set_node_id(guid::create_guid());
            _current_scene->create_node(node, mapToScene(pt));
        }
    }
}
void QtFlowDiagramView::scene_cleared()
{
    _selected_pin = nullptr;
    _temp_pin = nullptr;
    _temp_connection = nullptr;
    _highlight_pin = nullptr;
}
