#include <QMetaType>

#ifdef USE_X11
#ifdef HAVE_QTX11EXTRAS
#include <QX11Info>
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
#ifdef USE_X11
    if (glxewInit() != GLEW_OK)
    {
        std::cerr << "setSyncToVBlank: failed to initialize GLXEW" << std::endl;
    }

    auto wid = m_glWidget->effectiveWinId();
    if (wid == 0)
    {
        std::cerr << "setSyncToVBlank: did not find ID of native window for QWidget" << std::endl;
        return;
    }

#ifdef HAVE_QTX11EXTRAS
    Display *dpy = QX11Info::display();
#else
    Display *dpy = nullptr;
#endif
    if (!dpy)
    {
        std::cerr << "setSyncToVBlank: did not find Display for application" << std::endl;
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
