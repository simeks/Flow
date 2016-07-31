#ifndef __MAIN_WINDOW_H__
#define __MAIN_WINDOW_H__

#include <QMainWindow>

class QtFlowDiagramView;
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void setup_ui();

    QtFlowDiagramView* _diagram_view;
    
private slots:
    void on_exit_triggered();
    void on_open_file_triggered();
    void on_save_file_triggered();

};

#endif // __MAIN_WINDOW_H__
