/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlEvent.h

#ifndef _VRMLEVENT_
#define _VRMLEVENT_

#include "VrmlField.h"

namespace vrml
{

class VRMLEXPORT VrmlEvent
{

public:
    VrmlEvent(double timeStamp, const char *eventName, const VrmlField *value)
        : d_timeStamp(timeStamp)
        , d_eventName(new char[strlen(eventName) + 1])
        , d_value(value->clone())
        , d_refCount(1)
    {
        strcpy(d_eventName, eventName);
    }

    ~VrmlEvent()
    {
        delete[] d_eventName;
        delete d_value;
    }

    // VrmlEvents are reference counted.
    // The reference counting is manual (that is, each user of a
    // VrmlEvent, such as the VrmlNode class, calls reference()
    // and dereference() explicitly). Should make it internal...

    // Add/remove references to a VrmlEvent. This is silly, as it
    // requires the users of VrmlEvent to do the reference/derefs...
    VrmlEvent *reference()
    {
        ++d_refCount;
        return this;
    }
    void dereference()
    {
        if (--d_refCount == 0)
            delete this;
    }

    double timeStamp()
    {
        return d_timeStamp;
    }

protected:
    double d_timeStamp;
    char *d_eventName;
    VrmlField *d_value;
    int d_refCount;
};
}
#endif //_VRMLEVENT_
