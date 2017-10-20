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

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>


#include <QMainWindow>
#include <QMenuBar>
#include <QApplication>
#include <QOpenGLWidget>
#include <QApplication>

#include <cassert>

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

bool WindowTypeQtPlugin::windowCreate(int i)
{
    auto &conf = *coVRConfig::instance();
    if (!qApp)
        new QApplication(coCommandLine::argc(), coCommandLine::argv());

    auto it = m_windows.find(i);
    if (it != m_windows.end())
    {
        std::cerr << "WindowTypeQt: already managing window no. " << i << std::endl;
        return false;
    }

    auto &win = m_windows[i];
    win.index = i;

    win.window = new QMainWindow();
    win.window->setGeometry(conf.windows[i].ox, conf.windows[i].oy, conf.windows[i].sx, conf.windows[i].sy);
    win.window->show();

#ifdef __APPLE__
    //auto menubar = new QMenuBar(nullptr);
    auto menubar = win.window->menuBar();
    menubar->setNativeMenuBar(false);
#else
    auto menubar = win.window->menuBar();
#endif
    menubar->show();
    win.view = new ui::QtView(menubar);
    cover->ui->addView(win.view);

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
    cover->ui->removeView(win.view);
    delete win.view;
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
