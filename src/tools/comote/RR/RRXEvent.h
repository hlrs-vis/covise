/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RRXEVENT_H
#define RRXEVENT_H

enum rrxevent_types
{
    RREV_NONE = 0,
    RREV_KEYPRESS,
    RREV_KEYRELEASE,
    RREV_BTNPRESS,
    RREV_BTNRELEASE,
    RREV_MOTION,
    RREV_WHEEL,
    RREV_RESIZE,
    RREV_TOUCHPRESS,
    RREV_TOUCHRELEASE,
    RREV_TOUCHMOVE,
};

/* from <osgGA/GUIEventAdapter> */
enum RRModKeyMask
{
    RRMODKEY_SHIFT = 0x1 | 0x2,
    RRMODKEY_CTRL = 0x4 | 0x8,
    RRMODKEY_ALT = 0x10 | 0x20,
    RRMODKEY_META = 0x40 | 0x80
};

struct rrxevent
{
    int type;
    float x;
    float y;
    int d1;
    int d2;

    rrxevent()
        : type(0)
        , x(0.0f)
        , y(0.0f)
        , d1(0)
        , d2(0)
    {
    }

    rrxevent(int type, float x, float y, int d1, int d2)
        : type(type)
        , x(x)
        , y(y)
        , d1(d1)
        , d2(d2)
    {
    }
};
#endif
