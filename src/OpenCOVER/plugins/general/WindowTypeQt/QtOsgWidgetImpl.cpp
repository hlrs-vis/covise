#include <QMetaType>

#ifdef USE_X11
#ifdef HAVE_QTX11EXTRAS
#include <QX11Info>
#endif
#if QT_VERSION >= QT_VERSION_CHECK(6, 2, 0)
#include <QGuiApplication>
#endif

#include <GL/glxew.h>
#undef Status
#undef Type
#undef None
#undef Bool
#undef Event
#undef KeyPress
#undef KeyRelease
#undef FocusIn
#undef FocusOut
#undef FontChange
#undef Expose
#undef CursorShape
#endif

#include "QtOsgWidget.h"

#include <QOpenGLWidget>
#include <iostream>

void QtGraphicsWindow::setSyncToVBlank(bool flag)
{
#if defined(USE_X11)
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cerr << "setSyncToVBlank: failed to initialize GLEW. Error: " << glewGetErrorString(err) << std::endl;
    }

    auto wid = m_glWidget->effectiveWinId();
    if (wid == 0)
    {
        std::cerr << "setSyncToVBlank: did not find ID of native window for QWidget" << std::endl;
        return;
    }

    Display *dpy = nullptr;

#if QT_VERSION >= QT_VERSION_CHECK(6, 2, 0)
    if (!qGuiApp) {
        std::cerr << "setSyncToVBlank: qGuiApp is nullptr" << std::endl;
        return;
    }

    // Detect session type
    QByteArray sessionType = qgetenv("XDG_SESSION_TYPE");
    if (sessionType == "x11") {
        auto x11App = qGuiApp->nativeInterface<QNativeInterface::QX11Application>();
        if (!x11App) {
            std::cerr << "setSyncToVBlank: QX11Application native interface is nullptr" << std::endl;
            return;
        }
        dpy = x11App->display();
    } else if (sessionType == "wayland") {
        std::cerr << "setSyncToVBlank: Wayland detected, VSync via GLX is not supported." << std::endl;
        return;
    } else {
        std::cerr << "setSyncToVBlank: Unknown session type: " << sessionType.constData() << std::endl;
        return;
    }
#else
    #ifdef HAVE_QTX11EXTRAS
        dpy = QX11Info::display();
    #endif
#endif

    if (!dpy)
    {
        std::cerr << "setSyncToVBlank: did not find Display for application" << std::endl;
        return;
    }

    int screenNumber = XScreenNumberOfScreen(XDefaultScreenOfDisplay(dpy));
    const char *s = glXQueryExtensionsString(dpy, screenNumber);
    if(s==nullptr)
    {
        std::cerr << "no extensions, probably running MESA" << std::endl;
        return;
    }

    if (!glXSwapIntervalEXT)
    {
        std::cerr << "setSyncToVBlank: no glXSwapIntervalEXT" << std::endl;
        return;
    }

    glXSwapIntervalEXT(dpy, wid, flag ? 1 : 0);
    std::cout << "setSyncToVBlank: " << (flag ? "enabled" : "disabled") << " for XScreen " << screenNumber << std::endl;
#endif
}
