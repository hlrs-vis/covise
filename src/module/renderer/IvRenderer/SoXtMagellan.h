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

//  -*- C++ -*-

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

#ifndef _SO_XT_SPACEBALL_
#define _SO_XT_SPACEBALL_

#include <X11/X.h>
#include <X11/extensions/XInput.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/devices/SoXtDevice.h>
#include <Inventor/events/SoMotion3Event.h>
#include <Inventor/events/SoSpaceballButtonEvent.h>

// C-api: prefix=SoXtSpball
class SoXtMagellan : public SoXtDevice
{
public:
    enum Mask
    {
        MOTION = 0x01,
        PRESS = 0x02,
        RELEASE = 0x04,
        ALL = 0x07
    };

    //
    // valid event mask values:
    //	    SoXtMagellan::MOTION   - spaceball translation and rotation
    //	    SoXtMagellan::PRESS    - spaceball button press
    //	    SoXtMagellan::RELEASE  - spaceball button release
    //	    SoXtMagellan::ALL	    - all spaceball events
    // Bitwise OR these to specify whichEvents this device should queue.
    //
    // The second constructor allows the spaceball to be attached
    // to a different display than the one used by SoXt::init().

    SoXtMagellan();
    virtual ~SoXtMagellan();

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

    // the spaceball reports rotations and translations as integers.
    // these values must be scaled to be useful. these methods allow
    // the scale values to be set.
    // default values are .006 for translation and .006 for scale.
    // C-api: name=setRotScaleFactor
    virtual void setRotationScaleFactor(float f)
    {
        rotScale = f;
    }
    // C-api: name=getRotScaleFactor
    virtual float getRotationScaleFactor() const
    {
        return rotScale;
    }
    // C-api: name=setXlateScaleFactor
    virtual void setTranslationScaleFactor(float f)
    {
        transScale = f;
    }
    // C-api: name=getXlateScaleFactor
    virtual float getTranslationScaleFactor() const
    {
        return transScale;
    }

    // Return whether or not the spaceball device exists for use.
    // Method with no argument checks on the primary display.
    virtual SbBool exists()
    {
        return existsD(SoXt::getDisplay());
    }
    // C-api: name=existsD
    virtual SbBool existsD(Display *d);
    virtual void init(Display *d, Mask mask);

    // scale factors
    float rotScale;
    float transScale;

protected:
    Mask eventMask; // X event interest for this device
    SoMotion3Event *motionEvent; // spaceball rotation/translation
    SoSpaceballButtonEvent *buttonEvent; // spball button press/release

    // these event types are retrieved from the X server at run time
    int motionEventType;
    int buttonPressEventType;
    int buttonReleaseEventType;

    // event classes passed to XSelectExtensionEvent
    XEventClass eventClasses[3]; // max of 3 event classes for this
    int numEventClasses; // actual number we will queue
    int eventTypes[3]; // max of 3 event types for this

    // device id is set at runtime
    XDevice *device;

    // event translators!
    SoMotion3Event *translateMotionEvent(XDeviceMotionEvent *me);
    SoSpaceballButtonEvent *translateButtonEvent(XDeviceButtonEvent *be, SoButtonEvent::State whichState);
};

#endif /* _SO_XT_SPACEBALL_ */
