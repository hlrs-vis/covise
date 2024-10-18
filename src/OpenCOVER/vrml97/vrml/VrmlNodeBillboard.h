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
    static void initFields(VrmlNodeBillboard *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeBillboard(VrmlScene *, const std::string &name = "");

    virtual void render(Viewer *);

    virtual void accumulateTransform(VrmlNode *);
    virtual VrmlNode *getParentTransform();
    virtual void inverseTransform(Viewer *);
    virtual void inverseTransform(double *mat);

private:
    VrmlSFVec3f d_axisOfRotation;

    VrmlNode *d_parentTransform;
    Viewer::Object d_xformObject;
};
}
#endif //_VRMLNODEBILLBOARD_
