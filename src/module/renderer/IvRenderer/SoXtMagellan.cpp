/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *
 *  Copyright (C) 2000 Silicon Graphics, Inc.  All Rights Reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  Further, this software is distributed without any warranty that it is
 *  free of the rightful claim of any third person regarding infringement
 *  or the like.  Any license provided herein, whether implied or
 *  otherwise, applies only to this software file.  Patent licenses, if
 *  any, provided herein do not apply to combinations of this program with
 *  other software, or any other product whatsoever.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  Contact information: Silicon Graphics, Inc., 1600 Amphitheatre Pkwy,
 *  Mountain View, CA  94043, or:
 *
 *  http://www.sgi.com
 *
 *  For further information regarding this notice, see:
 *
 *  http://oss.sgi.com/projects/GenInfo/NoticeExplan/
 *
 */

/*
 * Copyright (C) 1990,91,92   Silicon Graphics, Inc.
 *
 _______________________________________________________________________
 ______________  S I L I C O N   G R A P H I C S   I N C .  ____________
 |
 |   $Revision: 1.1.1.1 $
 |
 |   Classes:
 |	SoXtMagellan
 |
|   Author(s): David Mott
|
______________  S I L I C O N   G R A P H I C S   I N C .  ____________
_______________________________________________________________________
*/

#include <Inventor/SbLinear.h>
#include <Inventor/SbTime.h>
#include <Inventor/Xt/SoXt.h>
#include "SoXtMagellan.h"
#include <Inventor/errors/SoDebugError.h>

#include <X11/Xlib.h>
#include <X11/extensions/XI.h>

#include <config/CoviseConfig.h>

extern "C" {
XDeviceInfo *XListInputDevices(Display *, int *);
XDevice *XOpenDevice(Display *, XID);
int XSelectExtensionEvent(Display *, Window, XEventClass *, int);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   Constructor which uses the primary display (specified at SoXt::init).
//
// public
//
SoXtMagellan::SoXtMagellan()
//
////////////////////////////////////////////////////////////////////////
{
    motionEvent = new SoMotion3Event;
    buttonEvent = new SoSpaceballButtonEvent;

    // these are empirically good default values
    rotScale = .006;
    transScale = .006;

    // but still the users should be allowed to change it:
    //                    1.0 = Uwe's dafault values

    // users scale the factors
    float userRotFact = covise::coCoviseConfig::getFloat("COVER.Input.Spaceball.ScaleRotation", 1.0f);
    float userTransFact = covise::coCoviseConfig::getFloat("COVER.Input.Spaceball.ScaleTranslation", 1.0f);

    rotScale *= userRotFact;
    transScale *= userTransFact;
}

////////////////////////////////////////////////////////////////////////
//
// This is where the real constructor work gets done.
//
// private
//
void
SoXtMagellan::init(
    Display *display,
    SoXtMagellan::Mask whichEvents)
//
////////////////////////////////////////////////////////////////////////
{
    if (display == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("SoXtMagellan::SoXtMagellan()",
                           "display is NULL.\n");
#endif
        return;
    }

    eventMask = whichEvents;

    // get the list of input devices that are attached to the display now
    XDeviceInfoPtr list;
    int numDevices;

    list = (XDeviceInfoPtr)XListInputDevices(display, &numDevices);

    // now run through the list looking for the spaceball device
    device = NULL;
    for (int i = 0; i < numDevices; i++)
    {
        // Open the spaceball device - the device id is set at runtime.
        if (strcmp(list[i].name, "spaceball") == 0)
        {
            device = XOpenDevice(display, list[i].id);
        }
        else if (strcmp(list[i].name, "magellan") == 0)
        {
            device = XOpenDevice(display, list[i].id);
        }
    }

    // make sure we found the spaceball device
    if (device == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("SoXtMagellan::SoXtMagellan",
                           "Sorry there is no Spaceball attached to this display");
#endif
        return;
    }

    // query the event types and classes
    uint32_t eventClass;
    numEventClasses = 0;

    if (eventMask & SoXtMagellan::MOTION)
    {
        DeviceMotionNotify(device, motionEventType, eventClass);
        eventClasses[numEventClasses] = eventClass;
        eventTypes[numEventClasses] = motionEventType;
        numEventClasses++;
    }

    if (eventMask & SoXtMagellan::PRESS)
    {
        DeviceButtonPress(device, buttonPressEventType, eventClass);
        eventClasses[numEventClasses] = eventClass;
        eventTypes[numEventClasses] = buttonPressEventType;
        numEventClasses++;
    }

    if (eventMask & SoXtMagellan::RELEASE)
    {
        DeviceButtonRelease(device, buttonReleaseEventType, eventClass);
        eventClasses[numEventClasses] = eventClass;
        eventTypes[numEventClasses] = buttonReleaseEventType;
        numEventClasses++;
    }

#ifdef DEBUG
    if (numEventClasses == 0)
    {
        SoDebugError::postWarning("SoXtMagellan::SoXtMagellan",
                                  "eventMask is NULL. No spaceball events will be received");
    }
#endif
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   Constructor.
//
// public
//
SoXtMagellan::~SoXtMagellan()
//
////////////////////////////////////////////////////////////////////////
{
    delete motionEvent;
    delete buttonEvent;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   returns whether the spaceball device exists for use or not.
//
// static, public
//
SbBool
SoXtMagellan::existsD(Display *display)
//
////////////////////////////////////////////////////////////////////////
{
    // get the list of input devices that are attached to the display now
    XDeviceInfoPtr list;
    int numDevices;

    if (display == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("SoXtMagellan::exists()",
                           "display is NULL.\n");
#endif
        return FALSE;
    }

    list = (XDeviceInfoPtr)XListInputDevices(display, &numDevices);

    // now run through the list looking for the spaceball device
    int i;
    for (i = 0; (i < numDevices) && (strcmp(list[i].name, "spaceball") != 0) && (strcmp(list[i].name, "magellan") != 0); i++)
        ; // shut up and loop
    //fprintf(stderr," %s",list[i].name)
    //fprintf(stderr,"%s",list[i].name);
    // if we broke out of the loop before i reached numDevices,
    // then the spaceball does in fact exist.
    return (i < numDevices);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   This selects input for spaceball device events which occur in w.
// The callback routine is proc, and the callback data is clientData.
//
// virtual public
//
void
SoXtMagellan::enable(
    Widget w,
    XtEventHandler proc,
    XtPointer clientData,
    Window window)
//
////////////////////////////////////////////////////////////////////////
{
    if (numEventClasses == 0)
        return;

    if (w == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("SoXtMagellan::enable",
                           "widget is NULL.");
#endif
        return;
    }

    if (window == 0)
    {
#ifdef DEBUG
        SoDebugError::post("SoXtMagellan::enable",
                           "widget must be realized (Window is NULL).");
#endif
        return;
    }

    Display *display = XtDisplay(w);
    if (display == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("SoXtMagellan::enable()",
                           "display is NULL.\n");
#endif
        return;
    }

    // select extension events for the spaceball which the user wants
    XSelectExtensionEvent(display, window, eventClasses, numEventClasses);

    // tell Inventor about these extension events!
    for (int i = 0; i < numEventClasses; i++)
        SoXt::addExtensionEventHandler(w, eventTypes[i], proc, clientData);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   This unselects input for spaceball device events which occur in w,
// i.e. spaceball events will no longer be recognized.
//
// virtual public
//
void
SoXtMagellan::disable(
    Widget w,
    XtEventHandler proc,
    XtPointer clientData)
//
////////////////////////////////////////////////////////////////////////
{
    // tell Inventor to forget about these classes
    for (int i = 0; i < numEventClasses; i++)
        SoXt::removeExtensionEventHandler(w, eventTypes[i], proc, clientData);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   This returns an SoEvent for the passed X event, if the event
// was generated by the mouse device.
//
// virtual public
//
const SoEvent *
SoXtMagellan::translateEvent(XAnyEvent *xevent)
//
////////////////////////////////////////////////////////////////////////
{
    SoEvent *event = NULL;

    // see if this is a spaceball event
    if (xevent->type == motionEventType)
    {
        XDeviceMotionEvent *me = (XDeviceMotionEvent *)xevent;
        if (me->deviceid == device->device_id)
            event = translateMotionEvent(me);
    }
    else if (xevent->type == buttonPressEventType)
    {
        XDeviceButtonEvent *be = (XDeviceButtonEvent *)xevent;
        if (be->deviceid == device->device_id)
            event = translateButtonEvent(be, SoButtonEvent::DOWN);
    }
    else if (xevent->type == buttonReleaseEventType)
    {
        XDeviceButtonEvent *be = (XDeviceButtonEvent *)xevent;
        if (be->deviceid == device->device_id)
            event = translateButtonEvent(be, SoButtonEvent::UP);
    }

    return event;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   This returns an SoMotion3Event for the passed X event.
//
// private
//
SoMotion3Event *
SoXtMagellan::translateMotionEvent(XDeviceMotionEvent *me)
//
////////////////////////////////////////////////////////////////////////
{
    // spaceball period event? - ignore it
    if (me->first_axis == 6)
        return NULL;

    // unknown data? - ignore it
    if ((me->first_axis != 0) || (me->axes_count != 6))
        return NULL;

    // ok, set up a motion event
    setEventPosition(motionEvent, me->x, me->y);
    int32_t secs = me->time / 1000;
    motionEvent->setTime(SbTime(secs, 1000 * (me->time - 1000 * secs)));
    motionEvent->setShiftDown(me->state & ShiftMask);
    motionEvent->setCtrlDown(me->state & ControlMask);
    motionEvent->setAltDown(me->state & Mod1Mask);

    // get the translation data in a right handed system (flip z)
    int *sbdata = me->axis_data;
    SbVec3f trans(sbdata[0], sbdata[1], sbdata[2]);
    trans *= transScale;
    motionEvent->setTranslation(trans);

    // get the rotation data in a right handed system (flip z)
    SbVec3f axis;

    axis.setValue(float(sbdata[3]), float(sbdata[4]), float(sbdata[5]));
    axis *= rotScale;
    float angle = axis.length();
    axis.normalize();
    motionEvent->setRotation(SbRotation(axis, angle));

    return motionEvent;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   This returns an SoSpaceballButtonEvent for the passed X event.
//
// private
//
SoSpaceballButtonEvent *
SoXtMagellan::translateButtonEvent(XDeviceButtonEvent *be,
                                   SoButtonEvent::State whichState)
//
////////////////////////////////////////////////////////////////////////
{
    setEventPosition(buttonEvent, be->x, be->y);
    int32_t secs = be->time / 1000;
    buttonEvent->setTime(SbTime(secs, 1000 * (be->time - 1000 * secs)));
    buttonEvent->setShiftDown(be->state & ShiftMask);
    buttonEvent->setCtrlDown(be->state & ControlMask);
    buttonEvent->setAltDown(be->state & Mod1Mask);

    // the value of be->button happens to match the SoSpaceballButton values
    buttonEvent->setButton((SoSpaceballButtonEvent::Button)be->button);
    buttonEvent->setState(whichState);

    return buttonEvent;
}
