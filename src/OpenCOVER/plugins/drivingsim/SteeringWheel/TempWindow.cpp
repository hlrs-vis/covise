/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TempWindow.h"
TemporaryWindow::TemporaryWindow()
    : handle_(0)
    , dc_(0)
    , context_(0)
    , instance_(0)
{
    create();
}

TemporaryWindow::TemporaryWindow(const TemporaryWindow &)
{
    throw "This is TemporaryWindow, please don't copy me!";
}

void TemporaryWindow::create()
{
    static int tempwnd_id__ = 0;
    std::ostringstream oss;
    oss << "tempwndow" << (++tempwnd_id__);
    classname_ = oss.str();

    instance_ = GetModuleHandle(0);

    WNDCLASS wndclass;

    wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wndclass.lpfnWndProc = DefWindowProc;
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;
    wndclass.hInstance = instance_;
    wndclass.hCursor = 0;
    wndclass.hIcon = 0;
    wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wndclass.lpszMenuName = 0;
    wndclass.lpszClassName = classname_.c_str();

    if (!RegisterClass(&wndclass))
    {
        fprintf(stderr, "RegisterClass Error:%d\n", GetLastError());
    }

    if (!(handle_ = CreateWindowEx(0,
                                   classname_.c_str(),
                                   TEXT(classname_.c_str()),
                                   WS_POPUP,
                                   0,
                                   0,
                                   100,
                                   100,
                                   0,
                                   0,
                                   instance_,
                                   0)))
    {
        kill();
        return;
    }
    fprintf(stderr, "handle:%d\n", handle_);

    if (!(dc_ = GetDC(handle_)))
    {
        kill();
        return;
    }

    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR),
        1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL,
        PFD_TYPE_RGBA,
        24,
        0, 0, 0, 0, 0, 0,
        0,
        0,
        0,
        0, 0, 0, 0,
        16,
        0,
        0,
        PFD_MAIN_PLANE,
        0,
        0, 0, 0
    };

    int visual_id = ChoosePixelFormat(dc_, &pfd);

    if (!SetPixelFormat(dc_, visual_id, &pfd))
    {
        kill();
        return;
    }

    if (!(context_ = wglCreateContext(dc_)))
    {
        kill();
        return;
    }
}

TemporaryWindow::~TemporaryWindow()
{
    kill();
}

void TemporaryWindow::kill()
{
    if (context_)
    {
        // mew 2005-05-09 commented out due to crashes.
        // possible causes are unsafe destructor ordering, or context already
        // deleted by window deletion; see:
        // http://openscenegraph.org/pipermail/osg-users/2005-May/052753.html
        //wglDeleteContext(context_);
        context_ = 0;
    }

    if (dc_)
    {
        ReleaseDC(handle_, dc_);
        dc_ = 0;
    }

    if (handle_)
    {
        DestroyWindow(handle_);
        handle_ = 0;
    }

    UnregisterClass(classname_.c_str(), instance_);
    instance_ = 0;
}

bool TemporaryWindow::makeCurrent()
{
    return wglMakeCurrent(dc_, context_);
}
