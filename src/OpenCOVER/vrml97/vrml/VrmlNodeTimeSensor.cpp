/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTimeSensor.cpp

#include "VrmlNodeTimeSensor.h"
#include "VrmlNodeType.h"

#include "MathUtils.h"

#include "VrmlScene.h"

using namespace vrml;

// TimeSensor factory. Add each TimeSensor to the scene for fast access.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeTimeSensor(scene);
}

// Define the built in VrmlNodeType:: "TimeSensor" fields

void VrmlNodeTimeSensor::initFields(VrmlNodeTimeSensor *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("cycleInterval", node->d_cycleInterval),
                     exposedField("enabled", node->d_enabled),
                     exposedField("loop", node->d_loop),
                     exposedField("startTime", node->d_startTime),
                     exposedField("stopTime", node->d_stopTime));
    if(t)
    {
        t->addEventOut("cycleTime", VrmlField::SFTIME);
        t->addEventOut("fraction_changed", VrmlField::SFFLOAT);
        t->addEventOut("isActive", VrmlField::SFBOOL);
        t->addEventOut("time", VrmlField::SFTIME);
    }                     

}

const char *VrmlNodeTimeSensor::name() { return "TimeSensor"; }

VrmlNodeTimeSensor::VrmlNodeTimeSensor(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_cycleInterval(1.0)
    , d_enabled(true)
    , d_loop(false)
    , d_startTime(0.0)
    , d_stopTime(0.0)
    , d_isActive(false)
    , d_lastTime(-1.0)
    , d_lastStartTime(-1.0)
    , fixedForTesting(false)
{
    oldFraction = 0;
    if (d_scene)
        d_scene->addTimeSensor(this);
    forceTraversal(false);
    if (getenv("VR_PREPARE_DEBUG_SNAPSHOTS_DIR"))
    {
        fixedForTesting = true;
    }
}

VrmlNodeTimeSensor::~VrmlNodeTimeSensor()
{
    if (d_scene)
        d_scene->removeTimeSensor(this);
}

VrmlNodeTimeSensor *VrmlNodeTimeSensor::toTimeSensor() const
{
    return (VrmlNodeTimeSensor *)this;
}

void VrmlNodeTimeSensor::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addTimeSensor(this);
}

//
// Generate timer events. If necessary, events prior to the timestamp (inTime)
// are generated to respect stopTimes and cycleIntervals. The timestamp
// should never be increased. This assumes the event loop delivers pending
// events in order (ascending time stamps). Should inTime be modified?
// Should ensure continuous events are delivered before discrete ones
// (such as cycleTime, isActive).

void VrmlNodeTimeSensor::update(VrmlSFTime &inTime)
{

    //VrmlSFTime timeNow( inTime );
    if (fixedForTesting)
    {
        d_time.set(0.5);
    }
    else
    {
        d_time.set(inTime.get());
    }

    if (d_enabled.get())
    {
        if (d_lastTime > inTime.get())
            d_lastTime = inTime.get();

        // Become active at startTime if either the valid stopTime hasn't
        // passed or we are looping.
        if (!d_isActive.get() && d_startTime.get() <= d_time.get() && d_startTime.get() != d_lastStartTime && ((d_stopTime.get() < d_startTime.get() || d_stopTime.get() > d_time.get()) || d_loop.get()))
        {
            d_isActive.set(true);
            d_lastStartTime = d_startTime.get();
            System::the->debug("TimeSensor.%s isActive TRUE\n", name());

            // Start at first tick >= startTime
            eventOut(d_time.get(), "isActive", d_isActive);

            eventOut(d_time.get(), "time", d_time);
            d_fraction.set(0.0);
            eventOut(d_time.get(), "fraction_changed", d_fraction);
            eventOut(d_time.get(), "cycleTime", d_time);
        }

        // Running (active and enabled)
        else if (d_isActive.get())
        {
            double f, cycleInt = d_cycleInterval.get();
            bool deactivate = false;

            // Are we done? Choose min of stopTime or start + single cycle.
            if ((d_stopTime.get() > d_startTime.get() && d_stopTime.get() <= d_time.get()) || ((!d_loop.get()) && d_startTime.get() + cycleInt <= d_time.get()))
            {
                d_isActive.set(false);

                // Must respect stopTime/cycleInterval exactly
                if ((d_startTime.get() + cycleInt < d_stopTime.get()) || (d_stopTime.get() < d_startTime.get()))
                    d_time = d_startTime.get() + cycleInt;
                else
                    d_time = d_stopTime;
                //if(d_lastTime > d_time.get()) // otherwise, we forget to start if startTime is set to d_time
                //d_lastTime = d_time.get();

                deactivate = true;
                d_startTime = 0.0;
            }

            if (cycleInt > 0.0 && d_time.get() > d_startTime.get())
            {
                if (!d_loop.get())
                {
                    if ((d_time.get() - d_startTime.get()) > cycleInt)
                        f = cycleInt;
                    else
                        f = (d_time.get() - d_startTime.get());
                }
                else
                {
                    f = fmod(d_time.get() - d_startTime.get(), cycleInt);
                }
            }
            else
                f = 0.0;

            // Fraction of cycle message
            //VrmlSFFloat fraction_changed( FPZERO(f) ? 1.0 : (f / cycleInt) );
            d_fraction.set(FPZERO(f) ? 1.0f : (float)(f / cycleInt));
            eventOut(d_time.get(), "fraction_changed", d_fraction);

            // Current time message
            eventOut(d_time.get(), "time", d_time);
            // End of cycle message (this may not miss cycles anymore)
            if (d_fraction.get() < oldFraction)
            {
                eventOut(d_time.get(), "cycleTime", d_time);
            }
            oldFraction = d_fraction.get();

            if (deactivate)
                eventOut(d_time.get(), "isActive", d_isActive);
        }

        // Tell the scene this node needs quick updates while it is active.
        // Should check whether time, fraction_changed eventOuts are
        // being used, and set delta to cycleTime if not...
        if (d_isActive.get())
            d_scene->setDelta(0.0);

        d_lastTime = d_time.get(); //inTime.get();
    }
}

// Ignore set_cycleInterval & set_startTime when active, deactivate
// if set_enabled FALSE is received when active.

void VrmlNodeTimeSensor::eventIn(double timeStamp,
                                 const char *eventName,
                                 const VrmlField *fieldValue)
{
    const char *origEventName = eventName;
    if (strncmp(eventName, "set_", 4) == 0)
        eventName += 4;

    System::the->debug("TimeSensor.%s eventIn %s\n", name(), origEventName);

    // Ignore set_cycleInterval & set_startTime when active
    if (strcmp(eventName, "cycleInterval") == 0 || strcmp(eventName, "startTime") == 0)
    {
        if (!d_isActive.get())
        {
            setFieldByName(eventName, *fieldValue);
            char eventOutName[256];
            strcpy(eventOutName, eventName);
            strcat(eventOutName, "_changed");
            eventOut(timeStamp, eventOutName, *fieldValue);
        }
    }
    else if (strcmp(eventName, "stopTime") == 0)
    {
        setFieldByName(eventName, *fieldValue);
        char eventOutName[256];
        strcpy(eventOutName, eventName);
        strcat(eventOutName, "_changed");
        eventOut(timeStamp, eventOutName, *fieldValue);
        if ((d_stopTime.get() > d_startTime.get() && d_stopTime.get() <= d_time.get()))
        {
            d_isActive.set(false);

            eventOut(d_time.get(), "isActive", d_isActive);
        }
    }

    // Shutdown if set_enabled FALSE is received when active
    else if (strcmp(eventName, "enabled") == 0)
    {
        setFieldByName(eventName, *fieldValue);
        if (d_isActive.get() && !d_enabled.get())
        {
            d_isActive.set(false);

            // Send relevant eventOuts (continuous ones first)
            //VrmlSFTime timeNow( timeStamp );
            d_time.set(timeStamp);
            eventOut(timeStamp, "time", d_time);

            double f, cycleInt = d_cycleInterval.get();
            if (cycleInt > 0.0)
                f = fmod(d_time.get() - d_startTime.get(), cycleInt);
            else
                f = 0.0;

            // Fraction of cycle message
            //VrmlSFFloat fraction_changed( FPZERO(f) ? 1.0 : (f / cycleInt) );
            d_fraction.set(FPZERO(f) ? 1.0f : (float)(f / cycleInt));

            eventOut(timeStamp, "fraction_changed", d_fraction);
            eventOut(timeStamp, "isActive", d_isActive);
        }
        else
        {
            // Become active if reenabled if either the valid stopTime hasn't
            // passed or we are looping.
            if (!d_isActive.get() && d_enabled.get() && d_startTime.get() <= d_time.get() && d_startTime.get() != d_lastStartTime && ((d_stopTime.get() < d_startTime.get() || d_stopTime.get() > d_time.get()) || d_loop.get()))
            {
                d_isActive.set(true);
                d_lastStartTime = d_startTime.get();
                System::the->debug("TimeSensor.%s isActive TRUE\n", name());

                // Start at first tick >= startTime
                eventOut(d_time.get(), "isActive", d_isActive);

                eventOut(d_time.get(), "time", d_time);
                d_fraction.set(0.0);
                eventOut(d_time.get(), "fraction_changed", d_fraction);
                eventOut(d_time.get(), "cycleTime", d_time);
            }
        }

        eventOut(timeStamp, "enabled_changed", *fieldValue);
    }

    // Let the generic code handle the rest.
    else
        VrmlNode::eventIn(timeStamp, origEventName, fieldValue);

    // TimeSensors shouldn't generate redraws.
    clearModified();
}
