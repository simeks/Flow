#include <Core/Common.h>

#include "MainWindow.h"
#include <QApplication>
#include <QSplashScreen>

#include <Core/Core.h>

#if defined(FLOW_BUILD_DEBUG) || !defined(FLOW_PLATFORM_WINDOWS)
int main(int argc, char** argv)
#else
int WinMain(HINSTANCE /*hInInstance*/, HINSTANCE /*hPrevInstance*/, char*, int /*nCmdShow*/)
#endif
{
    memory::initialize();
    int res;
    {
#if defined(FLOW_BUILD_DEBUG) || !defined(FLOW_PLATFORM_WINDOWS)
        QApplication a(argc, argv);
#else
        QApplication a(__argc, __argv);
#endif

        QSplashScreen* splash = new QSplashScreen();
        splash->setPixmap(QPixmap(":/res/splash.png"));
        splash->show();

        splash->showMessage("Initializing core...", Qt::AlignRight | Qt::AlignBottom, Qt::white);
        Core::create();
        // TODO: argc, argv
        Core::get().initialize(0, 0);

        MainWindow w;
        w.show();
        splash->finish(&w);

        delete splash;

        res = a.exec();

    }
    Core::destroy();
    memory::shutdown();
    return res;
}
