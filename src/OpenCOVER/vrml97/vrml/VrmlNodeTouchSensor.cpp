/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTouchSensor.cpp

#include "config.h"
#include "VrmlNodeTouchSensor.h"
#include "VrmlNodeType.h"

#include "MathUtils.h"
#include "System.h"
#include "Viewer.h"
#include "VrmlScene.h"

using namespace vrml;

// TouchSensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeTouchSensor(scene);
}

// Define the built in VrmlNodeType:: "TouchSensor" fields

VrmlNodeType *VrmlNodeTouchSensor::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("TouchSensor", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("hitNormal_changed", VrmlField::SFVEC3F);
    t->addEventOut("hitPoint_changed", VrmlField::SFVEC3F);
    t->addEventOut("hitTexCoord_changed", VrmlField::SFVEC2F);
    t->addEventOut("isOver", VrmlField::SFBOOL);
    t->addEventOut("isActive", VrmlField::SFBOOL);
    t->addEventOut("touchTime", VrmlField::SFTIME);

    return t;
}

VrmlNodeType *VrmlNodeTouchSensor::nodeType() const { return defineType(0); }

VrmlNodeTouchSensor::VrmlNodeTouchSensor(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
    , d_isActive(false)
    , d_isOver(false)
    , d_touchTime(0.0)
{
    setModified();
    forceTraversal(false);
}

VrmlNodeTouchSensor::~VrmlNodeTouchSensor()
{
}

VrmlNode *VrmlNodeTouchSensor::cloneMe() const
{
    return new VrmlNodeTouchSensor(*this);
}

VrmlNodeTouchSensor *VrmlNodeTouchSensor::toTouchSensor() const
{
    return (VrmlNodeTouchSensor *)this;
}

std::ostream &VrmlNodeTouchSensor::printFields(std::ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    return os;
}

// Doesn't compute the xxx_changed eventOuts yet...

void VrmlNodeTouchSensor::activate(double timeStamp,
                                   bool isOver, bool isActive,
                                   double *p)
{
    if (isOver && !isActive && d_isActive.get())
    {
        d_touchTime.set(timeStamp);
        eventOut(timeStamp, "touchTime", d_touchTime);
        //System::the->debug("TouchSensor.%s touchTime\n", name());
    }
    if (isOver)
    {
        float V[3] = { (float)p[0], (float)p[1], (float)p[2] };
        double M[16];
        inverseTransform(M);
        VM(V, M, V);
        d_hitPoint_changed.set(V[0], V[1], V[2]);
        eventOut(timeStamp, "hitPoint_changed", d_hitPoint_changed);
    }

    if (isOver != d_isOver.get())
    {
        d_isOver.set(isOver);
        eventOut(timeStamp, "isOver", d_isOver);
    }

    if (isActive != d_isActive.get())
    {
        d_isActive.set(isActive);
        eventOut(timeStamp, "isActive", d_isActive);
    }

    // if (isOver && any routes from eventOuts)
    //   generate xxx_changed eventOuts...
}

// Set the value of one of the node fields.

void VrmlNodeTouchSensor::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTouchSensor::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;

    return VrmlNodeChild::getField(fieldName);
}
