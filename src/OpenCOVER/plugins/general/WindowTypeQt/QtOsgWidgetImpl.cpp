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
#include <QOpenGLContext>
#include <QWindow>
#include <iostream>
#include <cstdlib>

void QtGraphicsWindow::setSyncToVBlank(bool flag)
{
    // Set NVIDIA-specific environment variable for VSync
    // This works with __NV_PRIME_RENDER_OFFLOAD and hybrid GPU setups
    if (flag) {
        putenv((char *)"__GL_SYNC_TO_VBLANK=1");
        std::cout << "setSyncToVBlank: Set __GL_SYNC_TO_VBLANK=1 (GPU global)" << std::endl;
    } else {
        putenv((char *)"__GL_SYNC_TO_VBLANK=0");
        std::cout << "setSyncToVBlank: Set __GL_SYNC_TO_VBLANK=0 (GPU global)" << std::endl;
    }

#if defined(USE_X11)
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
    if (sessionType == "x11" || sessionType == "tty") {
        auto x11App = qGuiApp->nativeInterface<QNativeInterface::QX11Application>();
        if (!x11App) {
            std::cerr << "setSyncToVBlank: QX11Application native interface is nullptr" << std::endl;
            return;
        }
        dpy = x11App->display();
    } else if (sessionType == "wayland") {
        // On Wayland, the environment variable should be sufficient
        // EGL swap interval control would require active EGL context which we don't have here
        std::cout << "setSyncToVBlank: Wayland detected, VSync control via __GL_SYNC_TO_VBLANK environment variable" << std::endl;
        std::cout << "setSyncToVBlank: Note: Actual VSync scheduling depends on Wayland compositor" << std::endl;
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
        std::cerr << "setSyncToVBlank: no GLX extensions, probably running MESA" << std::endl;
        return;
    }

    // Check for NVIDIA-specific extensions
    std::string extensions(s);
    bool hasNvidiaSwap = extensions.find("GLX_NV_swap_group") != std::string::npos;
    if (hasNvidiaSwap) {
        std::cout << "setSyncToVBlank: NVIDIA GPU detected (GLX_NV_swap_group present)" << std::endl;
    }

    if (!glXSwapIntervalEXT)
    {
        std::cerr << "setSyncToVBlank: no glXSwapIntervalEXT" << std::endl;
        return;
    }

    glXSwapIntervalEXT(dpy, wid, flag ? 1 : 0);
    std::cout << "setSyncToVBlank: " << (flag ? "enabled" : "disabled") 
              << " via GLX for XScreen " << screenNumber << std::endl;
#endif
}
