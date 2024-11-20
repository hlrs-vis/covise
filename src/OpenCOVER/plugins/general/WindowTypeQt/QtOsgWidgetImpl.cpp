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
#ifdef glxewInit
    if (glxewInit() != GLEW_OK)
    {
        std::cerr << "setSyncToVBlank: failed to initialize GLXEW" << std::endl;
    }
    else
#endif
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "setSyncToVBlank: failed to initialize GLEW" << std::endl;
    }

    auto wid = m_glWidget->effectiveWinId();
    if (wid == 0)
    {
        std::cerr << "setSyncToVBlank: did not find ID of native window for QWidget" << std::endl;
        return;
    }

    Display *dpy = nullptr;

    #if QT_VERSION >= QT_VERSION_CHECK(6, 2, 0)
        dpy = qGuiApp->nativeInterface<QNativeInterface::QX11Application>()->display();
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

    //  const char *s = glXQueryExtensionsString(dpy, wid);
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
#endif
}
