/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TEMP_WINDOW_H
#define TEMP_WINDOW_H

#include <util/common.h>

class TemporaryWindow
{
public:
    TemporaryWindow();
    TemporaryWindow(const TemporaryWindow &);
    ~TemporaryWindow();
    TemporaryWindow &operator=(const TemporaryWindow &)
    {
        return *this;
    }

    void create();
    void kill();

    HWND getHandle() const
    {
        return handle_;
    }
    HDC getDC() const
    {
        return dc_;
    }
    HGLRC getContext() const
    {
        return context_;
    }

    bool makeCurrent();

    HWND handle_;
    HDC dc_;
    HGLRC context_;
    HINSTANCE instance_;
    std::string classname_;
};

#endif
