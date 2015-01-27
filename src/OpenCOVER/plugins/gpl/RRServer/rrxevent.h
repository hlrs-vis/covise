#ifndef RREVENT_H
#define RREVENT_H

#include <X11/Xlib.h>

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

struct rrxevent
{
    int type;
    float x;
    float y;
    int d1;
    int d2;
};

bool translate(rrxevent *rrev, XEvent *xev, int *width, int *height);

#endif
