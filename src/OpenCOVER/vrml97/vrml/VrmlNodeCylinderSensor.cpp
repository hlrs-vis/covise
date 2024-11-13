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

void VrmlNodeCylinderSensor::initFields(VrmlNodeCylinderSensor *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("autoOffset", node->d_autoOffset),
                     exposedField("diskAngle", node->d_diskAngle),
                     exposedField("enabled", node->d_enabled),
                     exposedField("maxAngle", node->d_maxAngle),
                     exposedField("minAngle", node->d_minAngle),
                     exposedField("offset", node->d_offset));
    if (t)
    {
        t->addEventOut("isActive", VrmlField::SFBOOL);
        t->addEventOut("rotation_changed", VrmlField::SFROTATION);
        t->addEventOut("trackPoint_changed", VrmlField::SFVEC3F);
    }
    VrmlNodeChild::initFields(node, t);
}

const char *VrmlNodeCylinderSensor::name()
{
    return "CylinderSensor";
}


VrmlNodeCylinderSensor::VrmlNodeCylinderSensor(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
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
