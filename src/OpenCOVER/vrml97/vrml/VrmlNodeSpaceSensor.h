/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeSpaceSensor.h

#ifndef _VRMLNODESpaceSensor_
#define _VRMLNODESpaceSensor_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFRotation.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeSpaceSensor : public VrmlNodeChild
{

public:
    // Define the fields of SpaceSensor nodes
    static void initFields(VrmlNodeSpaceSensor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeSpaceSensor(VrmlScene *scene = 0);

    virtual VrmlNodeSpaceSensor *toSpaceSensor() const;

    void activate(double timeStamp, bool isActive, double *p, const double *M);

    virtual void accumulateTransform(VrmlNode *);
    virtual VrmlNode *getParentTransform();

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    // Fields
    VrmlSFBool d_autoOffset;
    VrmlSFBool d_enabled;
    VrmlSFBool d_rotationEnabled;
    VrmlSFVec3f d_maxPosition;
    VrmlSFVec3f d_minPosition;
    VrmlSFVec3f d_offset;

    VrmlSFBool d_isActive;
    VrmlSFVec3f d_translation;
    VrmlSFRotation d_rotation;
    VrmlSFVec3f d_trackPoint;

    VrmlSFVec3f d_activationPoint;

    VrmlNode *d_parentTransform;

    double invStartMove[16];
    double orientation[16];
    double startOrientation[16];
    double SM[16];
};
}
#endif //_VRMLNODESpaceSensor_
