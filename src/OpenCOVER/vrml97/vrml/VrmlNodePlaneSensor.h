/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePlaneSensor.h

#ifndef _VRMLNODEPLANESENSOR_
#define _VRMLNODEPLANESENSOR_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFVec2f.h"
#include "VrmlSFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodePlaneSensor : public VrmlNodeChild
{

public:
    // Define the fields of PlaneSensor nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodePlaneSensor(VrmlScene *scene = 0);
    virtual ~VrmlNodePlaneSensor();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodePlaneSensor *toPlaneSensor() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    void activate(double timeStamp, bool isActive, double *p);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

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
    VrmlSFVec2f d_maxPosition;
    VrmlSFVec2f d_minPosition;
    VrmlSFVec3f d_offset;

    VrmlSFBool d_isActive;
    VrmlSFVec3f d_translation;
    VrmlSFVec3f d_trackPoint;

    VrmlSFVec3f d_activationPoint;

    VrmlNode *d_parentTransform;
};
}
#endif //_VRMLNODEPLANESENSOR_
