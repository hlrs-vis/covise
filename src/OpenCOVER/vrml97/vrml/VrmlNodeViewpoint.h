/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeViewpoint.h

#ifndef _VRMLNODEVIEWPOINT_
#define _VRMLNODEVIEWPOINT_

#include "VrmlNode.h"
#include "VrmlField.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFRotation.h"
#include "VrmlSFString.h"
#include "VrmlSFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlScene;

class VRMLEXPORT VrmlNodeViewpoint : public VrmlNodeChild
{

public:
    bool lastBind;
    static void initFields(VrmlNodeViewpoint *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeViewpoint(VrmlScene *);
    virtual ~VrmlNodeViewpoint();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void accumulateTransform(VrmlNode *);
    virtual VrmlNode *getParentTransform();
    void recalcLast(); // save current position as last pos
    void setLastViewpointPosition(float *, float *);

    float fieldOfView()
    {
        return d_fieldOfView.get();
    }
    void getPosition(float *pos, float *ori);
    void getLastPosition(float *pos, float *ori);
    bool useLast()
    {
        if (!lastBind)
            return false;
        lastBind = false;
        return true;
    };

    const char *description()
    {
        return d_description.get() ? d_description.get() : "";
    }
    const char *type()
    {
        return d_type.get() ? d_type.get() : "free";
    }

private:
    VrmlSFFloat d_fieldOfView;
    VrmlSFBool d_jump;
    VrmlSFRotation d_orientation;
    VrmlSFVec3f d_position;
    VrmlSFVec3f d_centerOfRotation;
    VrmlSFRotation d_lastOrientation;
    VrmlSFRotation d_lastVPOrientation;
    VrmlSFVec3f d_lastPosition;
    VrmlSFVec3f d_lastVPPosition;
    VrmlSFString d_description;
    VrmlSFString d_type;

    VrmlNode *d_parentTransform;
    double startTime;
    float animFraction;
};
}
#endif //_VRMLNODEVIEWPOINT_
