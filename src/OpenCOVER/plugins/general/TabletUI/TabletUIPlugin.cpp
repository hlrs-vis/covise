/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: create windows with Qt
 **                                                                          **
 **                                                                          **
 ** Author: Martin Aumüller <aumueller@hlrs.de>
 **                                                                          **
\****************************************************************************/

#include "TabletUIPlugin.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/VRWindow.h>
#include <cover/ui/Manager.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/TabletView.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>

#include <QtGlobal>
#include <QMenuBar>
#include <QToolBar>
#include <QApplication>
#include <QLayout>
#include <QMessageBox>
#include <QDialog>
#include <QWindow>
#include <QFile>


#include <cassert>

#ifdef USE_X11
#ifdef HAVE_QTX11EXTRAS
#include <QX11Info>
#endif

#ifdef USE_X11_ICE
#include <X11/ICE/ICElib.h>
#endif
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#endif

#include <tui/TUIMainWindow.h>

using namespace opencover;

class TuiWindow: public TUIMainWindow
{
    TabletUIPlugin *plugin = nullptr;

public:
    TuiWindow(TabletUIPlugin *plugin): plugin(plugin) {}

    void notifyRemoveTabletUI() override
    {
        std::cerr << "TuiWindow::notifyRemoveTabletUI" << std::endl;
        if (plugin)
            plugin->destroy();
    }
};

TabletUIPlugin::TabletUIPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("QtWindow", cover->ui)
{
    //fprintf(stderr, "TabletUIPlugin::TabletUIPlugin\n");
}

// this is called if the plugin is removed at runtime
TabletUIPlugin::~TabletUIPlugin()
{
    if (m_ownsQApp)
    {
        qApp->quit();
        qApp->sendPostedEvents();
        qApp->processEvents();
        delete qApp;
    }
    //fprintf(stderr, "TabletUIPlugin::~TabletUIPlugin\n");
}

#ifdef USE_X11_ICE
static void iceIOErrorHandler(IceConn conn)
{
    (void)conn;
    std::cerr << "WindowTypeQt: ignoring ICE IO error" << std::endl;
}
#endif

bool TabletUIPlugin::init()
{
    if (!qApp)
    {
#ifdef USE_X11_ICE
        IceSetIOErrorHandler(&iceIOErrorHandler);
#endif
        QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
        QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
        new QApplication(coCommandLine::argc(), coCommandLine::argv());
        qApp->setWindowIcon(QIcon(":/icons/cover.ico"));
        //qApp->setAttribute(Qt::AA_PluginApplication);
        qApp->setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
        qApp->setAttribute(Qt::AA_DontCheckOpenGLContextThreadAffinity);
#ifdef __APPLE__
        qApp->setAttribute(Qt::AA_DontShowIconsInMenus);
#endif
        m_ownsQApp = true;
    }

#ifdef _WIN32
    std::cerr << "TabletUI plugin: socketpair not available on windows" << std::endl;
    return false;
#else
    int err = socketpair(AF_UNIX, SOCK_STREAM, 0, m_tuiFds);
    if (err != 0)
    {
        std::cerr << "TabletUI plugin: socketpair for internal tabletUI failed: " << strerror(errno) << std::endl;
        return false;
    }
    err = socketpair(AF_UNIX, SOCK_STREAM, 0, m_tuiSgFds);
    if (err != 0)
    {
        std::cerr << "TabletUI plugin: socketpair for internal tabletUI Scenegraph browser failed: " << strerror(errno)
                  << std::endl;
        return false;
    }
#endif

    for (auto fd: {m_tuiFds[0], m_tuiFds[1], m_tuiSgFds[0], m_tuiSgFds[1]})
    {
        cover->watchFileDescriptor(fd);
    }

    auto *tui = coTabletUI::instance();
    coTabletUI::ConfigData cd(m_tuiFds[0], m_tuiSgFds[0]);
    tui->reinit(cd);

    m_window = new TuiWindow(this);
    m_window->setFds(m_tuiFds[1], m_tuiSgFds[1]);
    m_window->setAttribute(Qt::WA_DeleteOnClose);
    m_window->show();

    tui->update();
    m_window->openServer();

    m_connection = QObject::connect(m_window, &QObject::destroyed, qApp,
                                    [this](QObject *)
                                    {
                                        if (m_connection)
                                        {
                                            QObject::disconnect(m_connection);
                                            m_connection = QMetaObject::Connection();
                                        }
                                        if (m_window)
                                        {
                                            m_window = nullptr;
                                        }
                                    });

    return true;
}

bool TabletUIPlugin::destroy()
{
    coTabletUI::instance()->init();

    for (auto fd: {m_tuiFds[0], m_tuiFds[1], m_tuiSgFds[0], m_tuiSgFds[1]})
    {
        if (fd != -1)
        {
            cover->unwatchFileDescriptor(fd);
            close(fd);
        }
    }

    if (m_connection)
    {
        QObject::disconnect(m_connection);
        m_connection = QMetaObject::Connection();
    }

    if (m_window)
    {
        auto win = m_window;
        m_window = nullptr;
        win->storeGeometry();
        win->close();
    }

    return true;
}

bool TabletUIPlugin::update()
{
    if (m_ownsQApp)
    {
        if (qApp)
            qApp->sendPostedEvents();
        if (qApp)
            qApp->processEvents();
    }
    if (!m_window)
        cover->removePlugin(this);
    return false;
}

COVERPLUGIN(TabletUIPlugin)
