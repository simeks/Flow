#include <Core/Common.h>

#include "MainWindow.h"
#include "ConsoleWidget.h"

#include "Flow/QtFlowDiagramScene.h"
#include "Flow/QtFlowDiagramView.h"
#include "Flow/QtNodePropertyWidget.h"

#include <QCoreApplication>
#include <QFileDialog>
#include <QSettings>

#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>

MainWindow::MainWindow(QWidget *parent) :
QMainWindow(parent)
{
    resize(1024, 768);

    QCoreApplication::setOrganizationName("MR24");
    QCoreApplication::setApplicationName("FlowLab");

    QSettings settings;

    settings.beginGroup("mainwindow");

    restoreGeometry(settings.value("geometry", saveGeometry()).toByteArray());
    restoreState(settings.value("savestate", saveState()).toByteArray());
    move(settings.value("pos", pos()).toPoint());
    resize(settings.value("size", size()).toSize());
    if (settings.value("maximized", isMaximized()).toBool())
        showMaximized();

    settings.endGroup();

    setup_ui();
}

MainWindow::~MainWindow()
{
    QSettings settings;

    settings.beginGroup("mainwindow");

    settings.setValue("geometry", saveGeometry());
    settings.setValue("savestate", saveState());
    settings.setValue("maximized", isMaximized());
    if (!isMaximized()) {
        settings.setValue("pos", pos());
        settings.setValue("size", size());
    }

    settings.endGroup();
}

void MainWindow::setup_ui()
{
    QAction* action_open_file = new QAction(this);
    action_open_file->setObjectName("open_file");
    QAction* action_save_file = new QAction(this);
    action_save_file->setObjectName("save_file");
    QAction* action_exit = new QAction(this);
    action_exit->setObjectName("exit");


    QMenuBar* menu_bar = new QMenuBar(this);
    setMenuBar(menu_bar);

    QMenu* menu_file = new QMenu(menu_bar);
    menu_bar->addAction(menu_file->menuAction());

    menu_file->addAction(action_open_file);
    menu_file->addAction(action_save_file);
    menu_file->addSeparator();
    menu_file->addAction(action_exit);

    _diagram_view = new QtFlowDiagramView(this);
    setCentralWidget(_diagram_view);

    setWindowTitle("MainWindow");
    menu_file->setTitle("File");
    action_open_file->setText("Open file");
    action_save_file->setText("Save file");
    action_exit->setText("Exit");

    QMetaObject::connectSlotsByName(this);
    QtNodePropertyWidget* property_widget = new QtNodePropertyWidget(this);
    property_widget->setObjectName("PropertyWidget");
    property_widget->setWindowTitle("Properties");
    addDockWidget(Qt::DockWidgetArea::RightDockWidgetArea, property_widget);

    ConsoleWidget* console_widget = new ConsoleWidget(this);
    console_widget->setObjectName("ConsoleWidget");
    console_widget->setWindowTitle("Console");
    addDockWidget(Qt::DockWidgetArea::BottomDockWidgetArea, console_widget);

    connect(_diagram_view, SIGNAL(flow_node_selected(QtBaseNode*)), property_widget, SLOT(flow_node_selected(QtBaseNode*)));
}

void MainWindow::on_exit_triggered()
{
    close();
}
void MainWindow::on_open_file_triggered()
{
    if (_diagram_view)
    {
        QtFlowDiagramScene* scene = _diagram_view->flow_scene();
        if (scene)
        {
            QString file_name = QFileDialog::getOpenFileName(this, "Open File", "", "Json (*.json)");
            if (file_name != "")
            {
                scene->load_from_file(file_name);
            }
        }
    }
}
void MainWindow::on_save_file_triggered()
{
    if (_diagram_view)
    {
        QtFlowDiagramScene* scene = _diagram_view->flow_scene();
        if (scene)
        {
            QString file_name = QFileDialog::getSaveFileName(this, "Save File", "", "Json (*.json)");
            if (file_name != "")
            {
                scene->save_to_file(file_name);
            }
        }
    }
}
