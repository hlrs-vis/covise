/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SO_XT_LINUX_MAGELLAN_
#define _SO_XT_LINUX_MAGELLAN_

#include "SoXtMagellan.h"
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

#define XHigh32(Value) (((Value) >> 16) & 0x0000FFFF)
#define XLow32(Value) ((Value)&0x0000FFFF)

#define MagellanInputMotionEvent 1
#define MagellanInputButtonPressEvent 2
#define MagellanInputButtonReleaseEvent 3

/** Linux driver for Magellan device.
  Requires X Windows driver from the Magellan distributor.
*/
class SoXtLinuxMagellan : public SoXtMagellan
{
public:
    SoXtLinuxMagellan();
    virtual ~SoXtLinuxMagellan();
    virtual void init(Display *d, Mask m);

protected:
    bool MagellanExist;

    // these functions will enable/disable this device for the passed widget.
    // the callback function f will be invoked when events occur in w.
    // data is the clientData which will be passed.
    virtual void enable(Widget w, XtEventHandler f, XtPointer data, Window win = 0);
    virtual void disable(Widget w, XtEventHandler f, XtPointer data);

    //
    // this converts an X event into an SoEvent,
    // returning NULL if the event is not from this device.
    //
    // C-api: name=xlateEv
    virtual const SoEvent *translateEvent(XAnyEvent *xevent);

    // Return whether or not the spaceball device exists for use.
    // Method with no argument checks on the primary display.
    SbBool exists()
    {
        return MagellanExist;
    }
    Atom MagellanMotionEvent, MagellanButtonPressEvent, MagellanReleaseEvent;
    Atom MagellanCommandEvent;
    Atom MagellanButtonReleaseEvent;
    enum _CommandMessages_
    {
        NoCommandMessage,
        CommandMessageApplicationWindow = 27695,
        CommandMessageApplicationSensitivity
    };
    union _MagellanTypeConversion_
    {
        float Float;
        short Short[2];
    };
    typedef union _MagellanTypeConversion_ MagellanTypeConversion;

    struct _MagellanFloatEvent_
    {
        int MagellanType;
        int MagellanButton;
        double MagellanData[6];
        int MagellanPeriod;
    };
    typedef struct _MagellanFloatEvent_ MagellanFloatEvent;

    enum _MagellanData_
    {
        MagellanX,
        MagellanY,
        MagellanZ,
        MagellanA,
        MagellanB,
        MagellanC
    };

    Window MagellanWindow; /* Magellan Driver Window */

    int MagellanSetWindow(Display *display, Window window);
    int MagellanTranslateEvent(Display *, XEvent *, MagellanFloatEvent *, double, double);

    static int MagellanErrorHandler(Display *display, XErrorEvent *Error);

    // event translators!
    SoMotion3Event *translateMotionEvent(XDeviceMotionEvent *me);
    SoSpaceballButtonEvent *translateButtonEvent(XDeviceButtonEvent *be, SoButtonEvent::State whichState);
};
#endif
