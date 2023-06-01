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
#include <cover/VRWindow.h>
#include <cover/ui/Manager.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>

#include <QMenuBar>
#include <QToolBar>
#include <QApplication>
#include <QOpenGLWidget>
#include <QLayout>
#include <QMessageBox>
#include <QDialog>
#include <QWindow>
#include <QFile>

#include "ui_AboutDialog.h"

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

using namespace opencover;

namespace {

bool enableCompositing(QWidget *window, bool state)
{
#ifdef USE_X11
    auto wid = window->effectiveWinId();
    if (wid == 0) {
        std::cerr << "enableCompositing: did not find ID of native window for QWidget" << std::endl;
        return false;
    }

#ifdef HAVE_QTX11EXTRAS
    Display *dpy = QX11Info::display();
#else
    Display *dpy = nullptr;
#endif
    if (!dpy) {
        std::cerr << "enableCompositing: did not find Display for application" << std::endl;
        return false;
    }

    Atom bypasscomp = XInternAtom(dpy, "_NET_WM_BYPASS_COMPOSITOR", False);

    long bypasscomp_on = state ? 2 : 1;
    if (state) {
        XChangeProperty(dpy, wid, bypasscomp, XA_CARDINAL, 32,
                        PropModeReplace, (unsigned char *)&bypasscomp_on, 1);
        bypasscomp_on = 0;
    }
    XChangeProperty(dpy, wid, bypasscomp, XA_CARDINAL, 32,
                    PropModeReplace, (unsigned char *)&bypasscomp_on, 1);
#endif
    return true;
}

}

WindowTypeQtPlugin::WindowTypeQtPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("QtWindow", cover->ui)
{
    //fprintf(stderr, "WindowTypeQtPlugin::WindowTypeQtPlugin\n");
}

// this is called if the plugin is removed at runtime
WindowTypeQtPlugin::~WindowTypeQtPlugin()
{
    //fprintf(stderr, "WindowTypeQtPlugin::~WindowTypeQtPlugin\n");
}

bool WindowTypeQtPlugin::init()
{
    bool useToolbar = covise::coCoviseConfig::isOn("toolbar", "COVER.UI.Qt", true);
    if (useToolbar)
    {
        m_toggleToolbar = new ui::Button("ToggleToolbar", this);
        m_toggleToolbar->setVisible(false, ~ui::View::WindowMenu);
        m_toggleToolbar->setText("Show toolbar");
        m_toggleToolbar->setState(true);
        cover->viewOptionsMenu->add(m_toggleToolbar);
    }

    auto sh = new ui::Action("ShowKeyboardHelp", this);
    sh->setText("Show keyboard help");
    sh->setVisible(false);
    sh->setCallback([this](){
            showKeyboardCommands();
    });
    sh->addShortcut("h");
    sh->addShortcut("F1");
    cover->viewOptionsMenu->add(sh);

    return true;
}

bool WindowTypeQtPlugin::destroy()
{
    while (!m_windows.empty())
    {
        windowDestroy(m_windows.begin()->second.index);
    }
    return true;
}

void WindowTypeQtPlugin::showKeyboardCommands()
{
    if (!m_keyboardHelp)
    {
        m_keyboardHelp = new KeyboardHelp(cover->ui);
        QObject::connect(m_keyboardHelp, &QDialog::finished, [this](int result){delete m_keyboardHelp; m_keyboardHelp = nullptr;});
    }
    m_keyboardHelp->show();
}

#ifdef USE_X11_ICE
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

    if (m_initializing) {
        m_initializing = false;

        for (auto w: m_windows)
        {
            w.second.widget->setFixedSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
        }
    }

    return up;
}

bool WindowTypeQtPlugin::windowCreate(int i)
{
    auto &conf = *coVRConfig::instance();
    if (!qApp)
    {
        m_deleteQApp = true;
#ifdef USE_X11_ICE
        IceSetIOErrorHandler(&iceIOErrorHandler);
#endif
        QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
        QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
        new QApplication(coCommandLine::argc(), coCommandLine::argv());
        qApp->setWindowIcon(QIcon(":/icons/cover.ico"));
        //qApp->setAttribute(Qt::AA_PluginApplication);
        qApp->setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
#if QT_VERSION >= QT_VERSION_CHECK(5, 8, 0)
        qApp->setAttribute(Qt::AA_DontCheckOpenGLContextThreadAffinity);
#endif
#ifdef __APPLE__
        qApp->setAttribute(Qt::AA_DontShowIconsInMenus);
#endif
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
    win.window->move(conf.windows[i].ox, conf.windows[i].oy);
    //win.window->resize(conf.windows[i].sx, conf.windows[i].sy);
    if (i > 0)
        win.window->setWindowTitle(("COVER"+std::to_string(i)).c_str());
    else
        win.window->setWindowTitle("COVER");
    win.window->setWindowIcon(QIcon(":/icons/cover.ico"));
    auto f = win.window->windowFlags();
    f |= Qt::CustomizeWindowHint;
    f |= Qt::WindowFullscreenButtonHint;
    win.window->setWindowFlags(f);
    win.window->show();
    window->connect(win.window, &QtMainWindow::closing, [this](){
        OpenCOVER::instance()->requestQuit();
    });
    window->connect(win.window, &QtMainWindow::fullScreenChanged, [this](bool state){
        m_update = true;
        VRWindow::instance()->makeFullScreen(state);
    });

    win.toggleFullScreen = new QAction(window);
    win.toggleFullScreen->setCheckable(true);
    win.toggleFullScreen->setChecked(false);
    win.toggleFullScreen->setText("Full Screen");
    window->connect(win.toggleFullScreen, &QAction::triggered, [this](bool state){
        m_update = true;
        VRWindow::instance()->makeFullScreen(state);
    });
    window->addContextAction(win.toggleFullScreen);

    win.toggleMenu = new QAction(window);
    win.toggleMenu->setCheckable(true);
    win.toggleMenu->setChecked(true);
    win.toggleMenu->setText("VR Menu");
    window->connect(win.toggleMenu, &QAction::triggered, [this](bool state){
        m_update = true;
        VRSceneGraph::instance()->setMenu(state);
    });
    window->addContextAction(win.toggleMenu);

    win.menubar = nullptr;
    win.nativeMenubar = nullptr;
#ifdef __APPLE__
    win.nativeMenubar = new QMenuBar(nullptr);
    win.nativeMenubar->setNativeMenuBar(true);
    win.nativeMenubar->show();
    bool nativeMenubar = covise::coCoviseConfig::isOn("nativeMenuBar", "COVER.UI.Qt", true);
    if (nativeMenubar)
    {
    }
    else
    {
        win.menubar = win.window->menuBar();
        win.menubar->setNativeMenuBar(false);
    }
#else
    win.menubar = win.window->menuBar();
#endif
    if (win.menubar)
        win.menubar->show();
    QToolBar *toolbar = nullptr;
    bool useToolbar = covise::coCoviseConfig::isOn("toolbar", "COVER.UI.Qt", true);
    if (useToolbar)
    {
        toolbar = new QToolBar("Toolbar", win.window);
        toolbar->layout()->setSpacing(2);
        toolbar->layout()->setContentsMargins(0, 0, 0, 0);
        win.window->addToolBar(toolbar);
        toolbar->show();
        window->addContextAction(toolbar->toggleViewAction());

        if (m_toggleToolbar)
        {
            m_toggleToolbar->setCallback([toolbar](bool state) { toolbar->toggleViewAction()->trigger(); });
            QObject::connect(toolbar->toggleViewAction(), &QAction::toggled,
                             [this](bool checked) { m_toggleToolbar->setState(checked); });
        }
    }
    win.toolbar = toolbar;
    if (win.menubar)
    {
        win.view.emplace_back(new ui::QtView(win.menubar, toolbar));
        cover->ui->addView(win.view.back());
    }
    if (win.nativeMenubar)
    {
        if (win.menubar)
            win.view.emplace_back(new ui::QtView(win.nativeMenubar));
        else
            win.view.emplace_back(new ui::QtView(win.nativeMenubar, toolbar));
        cover->ui->addView(win.view.back());
    }
#if 0
    win.view.emplace_back(new ui::QtView(toolbar));
    cover->ui->addView(win.view.back());
#endif

    window->connect(win.toggleFullScreen, &QAction::triggered, [this](bool state){
        m_update = true;
        VRWindow::instance()->makeFullScreen(state);
    });

    QAction *keyboardHelp = new QAction(window);
    keyboardHelp->setText("Keyboard commands...");
    window->connect(keyboardHelp, &QAction::triggered,
                    [this](bool)
                    {
                        showKeyboardCommands();
                    });

    QAction *aboutQt = new QAction(window);
    aboutQt->setText("About Qt...");
    window->connect(aboutQt, &QAction::triggered, [this](bool) { QMessageBox::aboutQt(new QDialog, "Qt"); });

    QAction *about = new QAction(window);
    about->setText("About COVER...");
    window->connect(about, &QAction::triggered, [this](bool) { aboutCover(); });

    unsigned idx = 0;
    for (auto *menubar: {win.menubar, win.nativeMenubar})
    {
        if (!menubar)
            continue;

        QMenu *helpMenu = new QMenu(menubar);
        helpMenu->setTearOffEnabled(true);
        helpMenu->setTitle("Help");
        menubar->addMenu(helpMenu);

        helpMenu->addAction(keyboardHelp);
        helpMenu->addSeparator();
        helpMenu->addAction(aboutQt);
        helpMenu->addAction(about);

        auto *view = win.view[idx];
        view->setInsertPosition(helpMenu->menuAction());
        ++idx;
    }

    QSurfaceFormat format;
    format.setVersion(2, 1);
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    //format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    //format.setOption(QSurfaceFormat::DebugContext);
    int bpc = covise::coCoviseConfig::getInt("bpc", "COVER.Framebuffer", 10);
    format.setRedBufferSize(bpc);
    format.setGreenBufferSize(bpc);
    format.setBlueBufferSize(bpc);
    int alpha = covise::coCoviseConfig::getInt("alpha", "COVER.Framebuffer", -1);
    if (alpha >= 0)
        format.setAlphaBufferSize(alpha);
    int depth = covise::coCoviseConfig::getInt("depth", "COVER.Framebuffer", 24);
    if (depth >= 0)
        format.setDepthBufferSize(depth);
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setStencilBufferSize(conf.numStencilBits());
    format.setStereo(conf.windows[i].stereo);

#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    bool found = false;
    bool sRGB = covise::coCoviseConfig::isOn("srgb", "COVER.Framebuffer", false, &found);
    if (!found)
        sRGB = covise::coCoviseConfig::isOn("COVER.FramebufferSRGB", false);
    if (sRGB)
    {
        std::cerr << "Enable GL_FRAMEBUFFER_SRGB" << std::endl;
        format.setColorSpace(QSurfaceFormat::sRGBColorSpace);
    }
#endif
    QSurfaceFormat::setDefaultFormat(format);
    win.widget = new QtOsgWidget(win.window);
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    if (sRGB)
    {
        win.widget->setTextureFormat(GL_SRGB8_ALPHA8);
    }
    else if (bpc > 12)
    {
        win.widget->setTextureFormat(GL_RGBA16);
    }
    else if (bpc > 10)
    {
        win.widget->setTextureFormat(GL_RGBA12);
    }
    else if (bpc > 8)
    {
        win.widget->setTextureFormat(GL_RGB10_A2);
    }
#endif
    win.widget->setFixedSize(conf.windows[i].sx, conf.windows[i].sy);
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

    QString text("This is <a href='https://www.hlrs.de/solutions/types-of-computing/visualization/covise/#heading2'>COVER</a> version %1.%2-<a href='https://github.com/hlrs-vis/covise/commit/%3'>%3</a> compiled on %4 for %5.");
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
    //std::cerr << "WindowTypeQt: destroying window " << num << std::endl;
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
    qApp->sendPostedEvents();
    qApp->processEvents();
    m_windows.erase(it);

    if (m_deleteQApp && m_windows.empty())
    {
        qApp->quit();
        qApp->sendPostedEvents();
        qApp->processEvents();
        delete qApp;
    }
}

void WindowTypeQtPlugin::windowFullScreen(int num, bool state)
{
    auto it = m_windows.find(num);
    if (it == m_windows.end())
    {
        std::cerr << "WindowTypeQt: window no. " << num << " not managed by this plugin" << std::endl;
        return;
    }
    auto &win = it->second;

    auto &conf = *coVRConfig::instance();

    win.toggleFullScreen->setChecked(state);
    if (state == win.fullscreen)
        return;

    win.fullscreen = state;
    m_update = true;

    if (state) {
        win.flags = win.window->windowFlags();
        auto state = win.window->windowState();
        win.state = state & ~Qt::WindowFullScreen;
#if 0
        if (!(state & Qt::WindowFullScreen)) {
            win.x = win.window->x();
            win.y = win.window->y();
            win.w = win.window->width();
            win.h = win.window->height();
        }
#endif
        if (win.toolbar) {
            win.toolbarVisible = win.toolbar->isVisible();
            win.toolbar->hide();
        }

        if (win.menubar)
        {
            win.menubar->hide();
        }
        //win.window->setWindowFlag(Qt::FramelessWindowHint, true);
        win.window->showFullScreen();
        if (!(win.state & Qt::WindowFullScreen))
            win.window->setWindowState(win.state | Qt::WindowFullScreen);
    } else {
        if (win.menubar)
        {
            win.menubar->setNativeMenuBar(false);
            win.menubar->show();
        }
#ifdef __APPLE__
        win.window->setWindowFlags(win.flags);
        win.window->setWindowState(win.state);
#if 0
        win.window->move(win.x, win.y);
        win.window->resize(win.w, win.h);
#endif
#endif
        if (win.toolbar && win.toolbarVisible)
            win.toolbar->show();
        //win.window->showNormal();
    }

    enableCompositing(win.window, state);
}

COVERPLUGIN(WindowTypeQtPlugin)
