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

void VrmlNodeTouchSensor::initFields(VrmlNodeTouchSensor *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("enabled", node->d_enabled));
    if(t)
    {
        t->addEventOut("hitNormal_changed", VrmlField::SFVEC3F);
        t->addEventOut("hitPoint_changed", VrmlField::SFVEC3F);
        t->addEventOut("hitTexCoord_changed", VrmlField::SFVEC2F);
        t->addEventOut("isOver", VrmlField::SFBOOL);
        t->addEventOut("isActive", VrmlField::SFBOOL);
        t->addEventOut("touchTime", VrmlField::SFTIME);
    }                     
}

const char *VrmlNodeTouchSensor::name() { return "TouchSensor"; }

VrmlNodeTouchSensor::VrmlNodeTouchSensor(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_enabled(true)
    , d_isActive(false)
    , d_isOver(false)
    , d_touchTime(0.0)
{
    setModified();
    forceTraversal(false);
}

VrmlNodeTouchSensor *VrmlNodeTouchSensor::toTouchSensor() const
{
    return (VrmlNodeTouchSensor *)this;
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
