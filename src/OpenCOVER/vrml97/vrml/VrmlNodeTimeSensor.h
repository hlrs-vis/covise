/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTimeSensor.h

#ifndef _VRMLNODETIMESENSOR_
#define _VRMLNODETIMESENSOR_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFTime.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlScene;

class VRMLEXPORT VrmlNodeTimeSensor : public VrmlNodeChild
{

public:
    // Define the fields of TimeSensor nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTimeSensor(VrmlScene *scene = 0);
    virtual ~VrmlNodeTimeSensor();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeTimeSensor *toTimeSensor() const;

    virtual void addToScene(VrmlScene *s, const char *);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    void update(VrmlSFTime &now);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual double getCycleInterval()
    {
        return d_cycleInterval.get();
    }
    virtual bool getEnabled()
    {
        return d_enabled.get();
    }
    virtual bool getLoop()
    {
        return d_loop.get();
    }
    virtual double getStartTime()
    {
        return d_startTime.get();
    }
    virtual double getStopTime()
    {
        return d_stopTime.get();
    }

private:
    // Fields
    VrmlSFTime d_cycleInterval;
    VrmlSFBool d_enabled;
    VrmlSFBool d_loop;
    VrmlSFTime d_startTime;
    VrmlSFTime d_stopTime;
    VrmlSFFloat d_fraction;
    VrmlSFTime d_time;

    // Internal state
    VrmlSFBool d_isActive;
    double d_lastTime;
    double d_lastStartTime;
    float oldFraction;

    // set to fixed time for testing
    bool fixedForTesting;
};
}
#endif //_VRMLNODETIMESENSOR_
