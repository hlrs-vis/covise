/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCylinderSensor.h

#ifndef _VRMLNODECYLINDERSENSOR_
#define _VRMLNODECYLINDERSENSOR_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFRotation.h"
#include "VrmlSFVec3f.h"
#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeCylinderSensor : public VrmlNodeChild
{

public:
    static void initFields(VrmlNodeCylinderSensor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCylinderSensor(VrmlScene *scene = 0);
    virtual VrmlNodeCylinderSensor *toCylinderSensor() const override;

    void activate(double timeStamp, bool isActive, double *p);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    // Fields
    VrmlSFBool d_autoOffset;
    VrmlSFFloat d_diskAngle;
    VrmlSFBool d_enabled;
    VrmlSFFloat d_maxAngle;
    VrmlSFFloat d_minAngle;
    VrmlSFFloat d_offset;

    VrmlSFBool d_isActive;
    VrmlSFRotation d_rotation;
    VrmlSFVec3f d_trackPoint;

    VrmlSFVec3f d_activationPoint;
    VrmlSFVec3f d_centerPoint;
};
}
#endif //_VRMLNODECYLINDERSENSOR_
