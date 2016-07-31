#ifndef __CONSOLE_WIDGET_H__
#define __CONSOLE_WIDGET_H__

#include <QDockWidget>
#include <QThread>
#include <Core/Common.h>

class Core;
class QLineEdit;
class QTextEdit;

class CommandInvoker : public QObject
{
    Q_OBJECT;
public:
    CommandInvoker();
    ~CommandInvoker();

private:
    QThread _command_thread;

public slots:
    void invoke();

signals:
    void on_command_finished();
};

class ConsoleWidget : public QDockWidget
{
    Q_OBJECT

public:
    explicit ConsoleWidget(QWidget *parent = nullptr);
    ~ConsoleWidget();

    void write(uint32_t flags, const char* msg);

private:
    void setup_ui();

    QLineEdit* _input_box;
    QTextEdit* _output_box;

    CommandInvoker* _command_invoker;

private slots:
    void on_input();
    void on_console_out(quint32 flags, QString msg);
    void on_command_finished();

signals:
    void on_console_out_signal(quint32 flags, QString msg);
    void on_invoke_command();
};

#endif // __CONSOLE_WIDGET_H__
