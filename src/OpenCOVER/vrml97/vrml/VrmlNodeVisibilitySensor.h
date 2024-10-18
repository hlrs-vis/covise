/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeVisibilitySensor.h

#ifndef _VRMLNODEVISIBILITYSENSOR_
#define _VRMLNODEVISIBILITYSENSOR_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFRotation.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeVisibilitySensor : public VrmlNodeChild
{

public:
    
    static void initFields(VrmlNodeVisibilitySensor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeVisibilitySensor(VrmlScene *scene = 0);

    void render(Viewer *) override;

private:
    // Fields
    VrmlSFVec3f d_center;
    VrmlSFBool d_enabled;
    VrmlSFVec3f d_size;

    // Internal state
    VrmlSFBool d_isActive;
    VrmlSFTime d_enterTime;
    VrmlSFTime d_exitTime;
};
}
#endif //_VRMLNODEVISIBILITYSENSOR_
