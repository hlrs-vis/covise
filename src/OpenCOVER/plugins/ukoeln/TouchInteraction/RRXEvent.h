/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRXEvent.h
// Defines the rrxevent data type for RRZK_rrxvent messages used by the RRServer,
// TuioClient and TouchInteraction plugins. Required for src/tools/comote.

#ifndef RRXEVENT_H_76EC02BA1BA24d03989C9DCDAFC203E8
#define RRXEVENT_H_76EC02BA1BA24d03989C9DCDAFC203E8

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
    RREV_TOUCHMOVE
};

typedef rrxevent_types rrxevent_types;

enum RRModKeyMask /* from <osgGA/GUIEventAdapter> */
{
    RRMODKEY_SHIFT = 0x01 | 0x02,
    RRMODKEY_CTRL = 0x04 | 0x08,
    RRMODKEY_ALT = 0x10 | 0x20,
    RRMODKEY_META = 0x40 | 0x80
};

typedef enum RRModKeyMask RRModKeyMask;

// TODO: These __X are required for binary compatibility with COmote for Mac
union rrxevent
{
    struct
    { // Only type field needed here...
        int type;
        float x;
        float y;
        int d1;
        int d2;
    };
    struct Key
    {
        int type;
        float __0;
        float __1;
        int button;
        int modifiers;
    } key;
    struct Mouse
    {
        int type;
        float x;
        float y;
        int button;
        int modifiers;
    } mouse;
    struct Wheel
    {
        int type;
        float __0;
        float __1;
        int delta;
        int __2;
    } wheel;
    struct Resize
    {
        int type;
        float w; // Use int here
        float h;
        int __0;
        int __1;
    } resize;
    struct Touch
    {
        int type;
        float x;
        float y;
        int id;
        int __0;
    } touch;
};

typedef union rrxevent rrxevent;

#ifdef __cplusplus

inline rrxevent makeRREmptyEvent()
{
    rrxevent e;

    e.type = RREV_NONE;

    return e;
}

inline rrxevent makeRRKeyEvent(int type, int button, int modifiers = 0)
{
    rrxevent e;

    e.key.type = type;
    e.key.button = button;
    e.key.modifiers = modifiers;

    return e;
}

inline rrxevent makeRRKeyPressEvent(int button, int modifiers = 0)
{
    return makeRRKeyEvent(RREV_KEYPRESS, button, modifiers);
}

inline rrxevent makeRRKeyReleaseEvent(int button, int modifiers = 0)
{
    return makeRRKeyEvent(RREV_KEYRELEASE, button, modifiers);
}

inline rrxevent makeRRMouseEvent(int type, float x, float y, int button = 0, int modifiers = 0)
{
    rrxevent e;

    e.mouse.type = type;
    e.mouse.x = x;
    e.mouse.y = y;
    e.mouse.button = button;
    e.mouse.modifiers = modifiers;

    return e;
}

inline rrxevent makeRRMousePressEvent(float x, float y, int button, int modifiers = 0)
{
    return makeRRMouseEvent(RREV_BTNPRESS, x, y, button, modifiers);
}

inline rrxevent makeRRMouseMoveEvent(float x, float y, int button, int modifiers = 0)
{
    return makeRRMouseEvent(RREV_MOTION, x, y, button, modifiers);
}

inline rrxevent makeRRMouseReleaseEvent(float x, float y, int button, int modifiers = 0)
{
    return makeRRMouseEvent(RREV_BTNRELEASE, x, y, button, modifiers);
}

inline rrxevent makeRRWheelEvent(int delta)
{
    rrxevent e;

    e.wheel.type = RREV_WHEEL;
    e.wheel.delta = delta;

    return e;
}

inline rrxevent makeRRResizeEvent(int w, int h)
{
    rrxevent e;

    e.resize.type = RREV_RESIZE;
    e.resize.w = w;
    e.resize.h = h;

    return e;
}

inline rrxevent makeRRTouchEvent(int type, float x, float y, int id)
{
    rrxevent e;

    e.touch.type = type;
    e.touch.x = x;
    e.touch.y = y;
    e.touch.id = id;

    return e;
}

inline rrxevent makeRRTouchPressEvent(float x, float y, int id)
{
    return makeRRTouchEvent(RREV_TOUCHPRESS, x, y, id);
}

inline rrxevent makeRRTouchMoveEvent(float x, float y, int id)
{
    return makeRRTouchEvent(RREV_TOUCHMOVE, x, y, id);
}

inline rrxevent makeRRTouchReleaseEvent(float x, float y, int id)
{
    return makeRRTouchEvent(RREV_TOUCHRELEASE, x, y, id);
}

#endif // __cplusplus

#endif // RRXEVENT_H
