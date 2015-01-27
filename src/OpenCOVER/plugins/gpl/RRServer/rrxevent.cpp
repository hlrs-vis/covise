#include "rrxevent.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "GraphicsWindowX11.h"

bool translate(rrxevent *rrev, XEvent *xev, int *width, int *height)
{
    memset(rrev, '\0', sizeof(*rrev));
    switch (xev->type)
    {
    case ButtonPress:
    case ButtonRelease:
    {
        XButtonEvent *be = (XButtonEvent *)xev;
        if (be->button >= 4 && be->button <= 5)
        { // wheel
            if (xev->type == ButtonPress)
            {
                rrev->type = RREV_WHEEL;
                rrev->d1 = be->button;
                rrev->d2 = be->button == 4 ? -1 : 1;
            }
        }
        else
        {
            rrev->type = (xev->type == ButtonPress) ? RREV_BTNPRESS : RREV_BTNRELEASE;
            int state = be->state >> 8;
            if (xev->type == ButtonPress)
                state |= 1 << (be->button - 1);
            else
                state &= ~(1 << (be->button - 1));
            rrev->d1 = be->button;
            rrev->d2 = state;
        }
        rrev->x = be->x;
        rrev->y = *height - be->y;
    }
    break;
    case KeyPress:
    case KeyRelease:
    {
        XKeyEvent *ke = (XKeyEvent *)xev;
        rrev->type = (xev->type == KeyPress) ? RREV_KEYPRESS : RREV_KEYRELEASE;
        rrev->x = ke->x;
        rrev->y = *height - ke->y;

        unsigned char buffer_return[32];
        int bytes_buffer = 32;
        KeySym keysym_return;

        int numChars = XLookupString(ke, reinterpret_cast<char *>(buffer_return), bytes_buffer, &keysym_return, NULL);
        int keySymbol = keysym_return;
#ifdef RR_COVISE
        if (!remapExtendedX11Key(keySymbol) && (numChars == 1))
#endif
        {
            keySymbol = buffer_return[0];
        }

        rrev->d1 = ke->state;
        rrev->d2 = keySymbol;
    }
    break;
    case MotionNotify:
    {
        XMotionEvent *me = (XMotionEvent *)xev;
        rrev->type = RREV_MOTION;
        rrev->x = me->x;
        rrev->y = *height - me->y;
    }
    break;
    case ResizeRequest:
    {
        XResizeRequestEvent *rre = (XResizeRequestEvent *)xev;
        rrev->type = RREV_RESIZE;
        rrev->x = rre->width;
        rrev->y = rre->height;
        *width = rre->width;
        *height = rre->height;
    }
    break;
    default:
        return false;
    }

    return true;
}
