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

#include "WindowTypeQt.h"
#include "QtView.h"
#include "QtOsgWidget.h"
#include "QtMainWindow.h"

#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>

#include <QMenuBar>
#include <QToolBar>
#include <QApplication>
#include <QOpenGLWidget>
#include <QApplication>
#include <QLayout>

#include <cassert>

#if defined(Q_OS_UNIX) && !defined(Q_OS_MAC)
#define USE_X11
#include <X11/ICE/ICElib.h>
#endif

using namespace opencover;

WindowTypeQtPlugin::WindowTypeQtPlugin()
{
    //fprintf(stderr, "WindowTypeQtPlugin::WindowTypeQtPlugin\n");
}

// this is called if the plugin is removed at runtime
WindowTypeQtPlugin::~WindowTypeQtPlugin()
{
    //fprintf(stderr, "WindowTypeQtPlugin::~WindowTypeQtPlugin\n");
}

bool WindowTypeQtPlugin::destroy()
{
    while (!m_windows.empty())
    {
        windowDestroy(m_windows.begin()->second.index);
    }
    return true;
}

#ifdef USE_X11
static void iceIOErrorHandler(IceConn conn)
{
    (void)conn;
    std::cerr << "WindowTypeQt: ignoring ICE IO error" << std::endl;
}
#endif

bool WindowTypeQtPlugin::update()
{
    bool checked = VRSceneGraph::instance()->menuVisible();
    checked = coVRMSController::instance()->syncBool(checked);
    if (checked != VRSceneGraph::instance()->menuVisible())
        VRSceneGraph::instance()->setMenu(checked);
    for (auto w: m_windows)
    {
        w.second.toggleMenu->setChecked(checked);
    }
    bool up = m_update;
    m_update = false;
    return up;
}

bool WindowTypeQtPlugin::windowCreate(int i)
{
    auto &conf = *coVRConfig::instance();
    if (!qApp)
    {
#ifdef USE_X11
        IceSetIOErrorHandler(&iceIOErrorHandler);
#endif
        new QApplication(coCommandLine::argc(), coCommandLine::argv());
    }

    auto it = m_windows.find(i);
    if (it != m_windows.end())
    {
        std::cerr << "WindowTypeQt: already managing window no. " << i << std::endl;
        return false;
    }

    auto &win = m_windows[i];
    win.index = i;

    auto window = new QtMainWindow();
    win.window = window;
    win.window->setGeometry(conf.windows[i].ox, conf.windows[i].oy, conf.windows[i].sx, conf.windows[i].sy);
    win.window->show();

    win.toggleMenu = new QAction(window);
    win.toggleMenu->setCheckable(true);
    win.toggleMenu->setChecked(true);
    win.toggleMenu->setText("VR Menu");
    window->connect(win.toggleMenu, &QAction::triggered, [this](bool state){
        m_update = true;
        VRSceneGraph::instance()->setMenu(state);
    });
    window->addContextAction(win.toggleMenu);

#ifdef __APPLE__
    //auto menubar = new QMenuBar(nullptr);
    auto menubar = win.window->menuBar();
    menubar->setNativeMenuBar(false);
#else
    auto menubar = win.window->menuBar();
#endif
    menubar->show();
    QToolBar *toolbar = nullptr;
    bool useToolbar = covise::coCoviseConfig::isOn("toolbar", "COVER.UI.Qt", true);
    if (useToolbar)
    {
        toolbar = new QToolBar("Toolbar", win.window);
        toolbar->layout()->setSpacing(2);
        toolbar->layout()->setMargin(0);
        win.window->addToolBar(toolbar);
        toolbar->show();
        window->addContextAction(toolbar->toggleViewAction());
    }
    win.view.emplace_back(new ui::QtView(menubar, toolbar));
    cover->ui->addView(win.view.back());
#if 0
    win.view.emplace_back(new ui::QtView(toolbar));
    cover->ui->addView(win.view.back());
#endif

    QSurfaceFormat format;
    format.setVersion(2, 1);
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    //format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    //format.setOption(QSurfaceFormat::DebugContext);
    //format.setRedBufferSize(8);
    format.setAlphaBufferSize(8);
    format.setDepthBufferSize(24);
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setStencilBufferSize(conf.numStencilBits());
    format.setStereo(conf.windows[i].stereo);
    QSurfaceFormat::setDefaultFormat(format);
    win.widget = new QtOsgWidget(win.window);
    win.window->setCentralWidget(win.widget);
    win.widget->show();
    coVRConfig::instance()->windows[i].context = win.widget->graphicsWindow();
    //std::cerr << "window " << i << ": ctx=" << coVRConfig::instance()->windows[i].context << std::endl;

    return true;
}

void WindowTypeQtPlugin::windowCheckEvents(int num)
{
    if (qApp)
    {
        qApp->sendPostedEvents();
        qApp->processEvents();
    }
}

void WindowTypeQtPlugin::windowUpdateContents(int num)
{
    auto win = dynamic_cast<QtGraphicsWindow *>(coVRConfig::instance()->windows[num].window.get());
    if (win && win->widget())
    {
        win->widget()->update();
    }
}

void WindowTypeQtPlugin::windowDestroy(int num)
{
    auto it = m_windows.find(num);
    if (it == m_windows.end())
    {
        std::cerr << "WindowTypeQt: window no. " << num << " not managed by this plugin" << std::endl;
        return;
    }

    auto &conf = *coVRConfig::instance();
    conf.windows[num].context = nullptr;
    conf.windows[num].windowPlugin = nullptr;
    conf.windows[num].window = nullptr;

    auto &win = it->second;
    while (!win.view.empty())
    {
        cover->ui->removeView(win.view.back());
        delete win.view.back();
        win.view.pop_back();
    }
    delete win.widget;
    delete win.window;
    m_windows.erase(it);

    if (m_windows.empty())
    {
        qApp->quit();
        qApp->sendPostedEvents();
        qApp->processEvents();
        delete qApp;
    }
}

COVERPLUGIN(WindowTypeQtPlugin)
