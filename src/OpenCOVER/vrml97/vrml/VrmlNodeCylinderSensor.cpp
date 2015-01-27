/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCylinderSensor.cpp

#include "config.h"
#include "VrmlNodeCylinderSensor.h"
#include "VrmlNodeType.h"

#include "MathUtils.h"
#include "Viewer.h"
#include "VrmlScene.h"
#include <math.h>

using namespace vrml;

// CylinderSensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeCylinderSensor(scene);
}

// Define the built in VrmlNodeType:: "CylinderSensor" fields

VrmlNodeType *VrmlNodeCylinderSensor::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("CylinderSensor", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("autoOffset", VrmlField::SFBOOL);
    t->addExposedField("diskAngle", VrmlField::SFFLOAT);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("maxAngle", VrmlField::SFFLOAT);
    t->addExposedField("minAngle", VrmlField::SFFLOAT);
    t->addExposedField("offset", VrmlField::SFFLOAT);
    t->addEventOut("isActive", VrmlField::SFBOOL);
    t->addEventOut("rotation_changed", VrmlField::SFROTATION);
    t->addEventOut("trackPoint_changed", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeCylinderSensor::nodeType() const
{
    return defineType(0);
}

VrmlNodeCylinderSensor::VrmlNodeCylinderSensor(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_autoOffset(true)
    , d_diskAngle(0.262f)
    , d_enabled(true)
    , d_maxAngle(-1.0f)
    , d_minAngle(0.0f)
    , d_offset(0.0f)
    , d_isActive(false)
{
    setModified();
}

VrmlNodeCylinderSensor::~VrmlNodeCylinderSensor()
{
}

VrmlNode *VrmlNodeCylinderSensor::cloneMe() const
{
    return new VrmlNodeCylinderSensor(*this);
}

// mgiger 6/16/00
VrmlNodeCylinderSensor *VrmlNodeCylinderSensor::toCylinderSensor() const
{
    return (VrmlNodeCylinderSensor *)this;
}

std::ostream &VrmlNodeCylinderSensor::printFields(std::ostream &os, int indent)
{
    if (!d_autoOffset.get())
        PRINT_FIELD(autoOffset);
    if (!FPEQUAL(d_diskAngle.get(), 0.262))
        PRINT_FIELD(diskAngle);
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!FPEQUAL(d_maxAngle.get(), -1.0))
        PRINT_FIELD(maxAngle);
    if (!FPZERO(d_minAngle.get()))
        PRINT_FIELD(minAngle);
    if (!FPZERO(d_offset.get()))
        PRINT_FIELD(offset);

    return os;
}

const VrmlField *VrmlNodeCylinderSensor::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "autoOffset") == 0)
        return &d_autoOffset;
    else if (strcmp(fieldName, "diskAngle") == 0)
        return &d_diskAngle;
    else if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "maxAngle") == 0)
        return &d_maxAngle;
    else if (strcmp(fieldName, "minAngle") == 0)
        return &d_minAngle;
    else if (strcmp(fieldName, "offset") == 0)
        return &d_offset;

    // eventOuts
    else if (strcmp(fieldName, "isActive") == 0)
        return &d_isActive;
    else if (strcmp(fieldName, "rotation") == 0)
        return &d_rotation;
    else if (strcmp(fieldName, "trackPoint") == 0)
        return &d_trackPoint;

    return VrmlNodeChild::getField(fieldName);
}

// Set the value of one of the node fields.

void VrmlNodeCylinderSensor::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    if
        TRY_FIELD(autoOffset, SFBool)
    else if
        TRY_FIELD(diskAngle, SFFloat)
    else if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(maxAngle, SFFloat)
    else if
        TRY_FIELD(minAngle, SFFloat)
    else if
        TRY_FIELD(offset, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

void VrmlNodeCylinderSensor::activate(double timeStamp,
                                      bool isActive,
                                      double *p)
{ // Become active
    if (isActive && !d_isActive.get())
    {
        d_isActive.set(isActive);

        // set activation point in world coords
        d_activationPoint.set((float)p[0], (float)p[1], (float)p[2]);

        if (d_autoOffset.get())
        {
            d_rotation.get()[3] = d_offset.get();
            d_rotation.get()[0] = 0;
            d_rotation.get()[1] = 1;
            d_rotation.get()[2] = 0;
        }

        // calculate the center of the object in world coords
        float V[3] = { 0.0, 0.0, 0.0 };
        double M[16];
        inverseTransform(M);
        VM(V, M, V);
        d_centerPoint.set(V[0], V[1], V[2]);

        // send message
        eventOut(timeStamp, "isActive", d_isActive);
    }

    // Become inactive
    else if (!isActive && d_isActive.get())
    {
        d_isActive.set(isActive);
        eventOut(timeStamp, "isActive", d_isActive);

        // save auto offset of rotation
        if (d_autoOffset.get())
        {
            d_offset = d_rotation.get()[3];
            eventOut(timeStamp, "offset_changed", d_offset);
        }
    }

    // Tracking
    else if (isActive)
    {

        // get local coord for touch point
        float V[3] = { (float)p[0], (float)p[1], (float)p[2] };
        double M[16];
        inverseTransform(M);
        VM(V, M, V);
        d_trackPoint.set(V[0], V[1], V[2]);
        eventOut(timeStamp, "trackPoint_changed", d_trackPoint);

        float V2[3] = { (float)p[0], (float)p[1], (float)p[2] };
        float tempv[3];
        Vdiff(tempv, V2, d_centerPoint.get());
        VrmlSFVec3f dir1(tempv[0], tempv[1], tempv[2]);
        double dist = dir1.length(); // get the length of the pre-normalized vector
        dir1.normalize();
        Vdiff(tempv, d_activationPoint.get(), d_centerPoint.get());
        VrmlSFVec3f dir2(tempv[0], tempv[1], tempv[2]);
        dir2.normalize();

        Vcross(tempv, dir1.get(), dir2.get());
        VrmlSFVec3f cx(tempv[0], tempv[1], tempv[2]);

        VrmlSFRotation newRot(0, 1, 0, (float)(dist * acos(dir1.dot(&dir2))));
        if (d_autoOffset.get())
            newRot.get()[3] += d_offset.get();
        d_rotation = newRot;

        eventOut(timeStamp, "rotation_changed", d_rotation);
    }
}
