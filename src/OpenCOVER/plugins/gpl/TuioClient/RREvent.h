// RREvent.h

#ifndef RREVENT_H
#define RREVENT_H

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

    rrxevent()
        : type(RREV_NONE)
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
