/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// TouchCursor.h

#ifndef TOUCH_CURSOR_H
#define TOUCH_CURSOR_H

struct TouchCursor
{
    int id;
    float x;
    float y;

    TouchCursor()
        : id(-1)
        , x(0.0f)
        , y(0.0f)
    {
    }

    TouchCursor(int id, float x, float y)
        : id(id)
        , x(x)
        , y(y)
    {
    }

    TouchCursor(const TouchCursor *p)
        : id(p->id)
        , x(p->x)
        , y(p->y)
    {
    }
};

#endif
