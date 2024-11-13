/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeProximitySensor.h

#ifndef _VRMLNODEPROXIMITYSENSOR_
#define _VRMLNODEPROXIMITYSENSOR_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFRotation.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlScene;

class VRMLEXPORT VrmlNodeProximitySensor : public VrmlNodeChild
{

public:
    // Define the fields of ProximitySensor nodes
    static void initFields(VrmlNodeProximitySensor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeProximitySensor(VrmlScene *scene = 0);

    virtual void render(Viewer *);

    virtual void remoteEvent(double timeStamp,
                             bool isOver, bool isActive, float *point);

private:
    // Fields
    VrmlSFVec3f d_center;
    VrmlSFBool d_enabled;
    VrmlSFVec3f d_size;

    // Internal state
    VrmlSFBool d_isActive;
    VrmlSFVec3f d_position;
    VrmlSFRotation d_orientation;
    VrmlSFTime d_enterTime;
    VrmlSFTime d_exitTime;
};
}
#endif //_VRMLNODEPROXIMITYSENSOR_
