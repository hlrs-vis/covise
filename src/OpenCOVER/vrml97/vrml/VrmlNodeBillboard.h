/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBillboard.h

#ifndef _VRMLNODEBILLBOARD_
#define _VRMLNODEBILLBOARD_

#include "VrmlNodeGroup.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeBillboard : public VrmlNodeGroup
{

public:
    // Define the fields of Billboard nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeBillboard(VrmlScene *);
    virtual ~VrmlNodeBillboard();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void accumulateTransform(VrmlNode *);
    virtual VrmlNode *getParentTransform();
    virtual void inverseTransform(Viewer *);
    virtual void inverseTransform(double *mat);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFVec3f d_axisOfRotation;

    VrmlNode *d_parentTransform;
    Viewer::Object d_xformObject;
};
}
#endif //_VRMLNODEBILLBOARD_
