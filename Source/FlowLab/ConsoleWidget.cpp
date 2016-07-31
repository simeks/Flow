#include <Core/Common.h>
#include <Core/Core.h>
#include <Core/Flow/FlowSystem.h>

#include "ConsoleWidget.h"

#include <QLineEdit>
#include <QTextEdit>
#include <QVBoxLayout>

namespace
{
    void console_output(void* data, uint32_t flags, const char* msg)
    {
        ConsoleWidget* widget = (ConsoleWidget*)data;
        widget->write(flags, msg);
    }
}

CommandInvoker::CommandInvoker()
{
    this->moveToThread(&_command_thread);
    _command_thread.start();
}
CommandInvoker::~CommandInvoker()
{
    _command_thread.quit();
    _command_thread.wait(1000);
    if (_command_thread.isRunning())
    {
        _command_thread.terminate();
        _command_thread.wait();
    }
}

void CommandInvoker::invoke()
{
    for (int i = 0; i < 10; ++i)
    {
        Sleep(1000);
        console::print("%d\n", i);
    }
    emit on_command_finished();
}


ConsoleWidget::ConsoleWidget(QWidget *parent) :
QDockWidget(parent)
{
    setup_ui();

    connect(this, SIGNAL(on_console_out_signal(quint32, QString)), SLOT(on_console_out(quint32, QString)), Qt::QueuedConnection);
    console::set_callback(console_output, this);

    _command_invoker = new CommandInvoker();
    connect(this, SIGNAL(on_invoke_command()), _command_invoker, SLOT(invoke()));
    connect(_command_invoker, SIGNAL(on_command_finished()), this, SLOT(on_command_finished()));
}

ConsoleWidget::~ConsoleWidget()
{
    delete _command_invoker;

    console::set_callback(nullptr, nullptr);
}
void ConsoleWidget::write(uint32_t flags, const char* msg)
{
    emit on_console_out_signal(flags, msg);
}

void ConsoleWidget::setup_ui()
{
    setGeometry(QRect(240, 90, 120, 80));
    QWidget* content = new QWidget(this);
    setWidget(content);
    
    QVBoxLayout* layout = new QVBoxLayout(content);
    layout->setContentsMargins(2, 2, 2, 2);
    _output_box = new QTextEdit(content);
    _output_box->setReadOnly(true);
    _input_box = new QLineEdit(content);

    layout->addWidget(_output_box);
    layout->addWidget(_input_box);

    connect(_input_box, SIGNAL(returnPressed()), this, SLOT(on_input()));
}
void ConsoleWidget::on_input()
{
    _input_box->setEnabled(false);
    _input_box->setText("");

    emit on_invoke_command();
}
void ConsoleWidget::on_console_out(quint32, QString msg)
{
    _output_box->setText(_output_box->toPlainText() + msg);
}
void ConsoleWidget::on_command_finished()
{
    _input_box->setEnabled(true);
}
