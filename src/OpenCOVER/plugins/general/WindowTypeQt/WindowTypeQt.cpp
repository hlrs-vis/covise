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
#include "KeyboardHelp.h"

#include <config/CoviseConfig.h>
#include <util/covise_version.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <QMenuBar>
#include <QToolBar>
#include <QApplication>
#include <QOpenGLWidget>
#include <QApplication>
#include <QLayout>
#include <QMessageBox>
#include <QDialog>

#include "ui_AboutDialog.h"

#include <cassert>

#ifdef USE_X11
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
    window->connect(win.window, &QtMainWindow::closing, [this, i](){
        OpenCOVER::instance()->requestQuit();
    });

    win.toggleMenu = new QAction(window);
    win.toggleMenu->setCheckable(true);
    win.toggleMenu->setChecked(true);
    win.toggleMenu->setText("VR Menu");
    window->connect(win.toggleMenu, &QAction::triggered, [this](bool state){
        m_update = true;
        VRSceneGraph::instance()->setMenu(state);
    });
    window->addContextAction(win.toggleMenu);

    QMenuBar *menubar = nullptr;
#ifdef __APPLE__
    if (covise::coCoviseConfig::isOn("nativeMenuBar", "COVER.UI.Qt", false))
    {
        menubar = new QMenuBar(nullptr);
        menubar->setNativeMenuBar(true);
    }
    else
    {
        menubar = win.window->menuBar();
        menubar->setNativeMenuBar(false);
    }
#endif
    if (!menubar)
        menubar = win.window->menuBar();
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


    QMenu *helpMenu = new QMenu(menubar);
    helpMenu->setTearOffEnabled(true);
    helpMenu->setTitle("Help");
    menubar->addMenu(helpMenu);
    win.view.back()->setInsertPosition(helpMenu->menuAction());

    QAction *keyboardHelp = new QAction(helpMenu);
    helpMenu->addAction(keyboardHelp);
    keyboardHelp->setText("Keyboard commands...");
    window->connect(keyboardHelp, &QAction::triggered, [this](bool){
        if (!m_keyboardHelp)
        {
            m_keyboardHelp = new KeyboardHelp(cover->ui);
        }
        m_keyboardHelp->show();
    });

    helpMenu->addSeparator();

    QAction *aboutQt = new QAction(helpMenu);
    helpMenu->addAction(aboutQt);
    aboutQt->setText("About Qt...");
    window->connect(aboutQt, &QAction::triggered, [this](bool){
        QMessageBox::aboutQt(new QDialog, "Qt");
    });

    QAction *about = new QAction(helpMenu);
    helpMenu->addAction(about);
    about->setText("About COVER...");
    window->connect(about, &QAction::triggered, [this](bool){
        aboutCover();
    });

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

#if QT_VERSION >= 0x050A00
    bool sRGB = covise::coCoviseConfig::isOn("COVER.FramebufferSRGB", false);
    if (sRGB)
    {
        std::cerr << "Enable GL_FRAMEBUFFER_SRGB" << std::endl;
        format.setColorSpace(QSurfaceFormat::sRGBColorSpace);
    }
#endif
    QSurfaceFormat::setDefaultFormat(format);
    win.widget = new QtOsgWidget(win.window);
#if QT_VERSION >= 0x050A00
    if (sRGB)
    {
        win.widget->setTextureFormat(GL_SRGB8_ALPHA8);
    }
#endif
    win.window->setCentralWidget(win.widget);
    win.widget->show();
    conf.windows[i].context = win.widget->graphicsWindow();
    conf.windows[i].doublebuffer = false;

    //std::cerr << "window " << i << ": ctx=" << coVRConfig::instance()->windows[i].context << std::endl;

    qApp->sendPostedEvents();
    qApp->processEvents();

    return true;
}

void WindowTypeQtPlugin::aboutCover() const
{
    using covise::CoviseVersion;

    QDialog *aboutDialog = new QDialog;
    Ui::AboutDialog *ui = new Ui::AboutDialog;
    ui->setupUi(aboutDialog);

    QFile file(":/aboutData/README-3rd-party.txt");
    if (file.open(QIODevice::ReadOnly))
    {
        QString data(file.readAll());
        ui->textEdit->setPlainText(data);
    }

    QString text("This is <a href='http://www.hlrs.de/de/solutions-services/service-portfolio/visualization/covise/opencover/'>COVER</a> version %1.%2-<a href='https://github.com/hlrs-vis/covise/commit/%3'>%3</a> compiled on %4 for %5.");
    text = text.arg(CoviseVersion::year())
        .arg(CoviseVersion::month())
        .arg(CoviseVersion::hash())
        .arg(CoviseVersion::compileDate())
        .arg(CoviseVersion::arch());
    text.append("<br>Follow COVER and COVISE developement on <a href='https://github.com/hlrs-vis/covise'>GitHub</a>!");

    ui->label->setText(text);
    ui->label->setTextInteractionFlags(Qt::TextBrowserInteraction | Qt::LinksAccessibleByMouse);
    ui->label->setOpenExternalLinks(true);

    aboutDialog->exec();
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
