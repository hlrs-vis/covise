/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTransform.h

#ifndef _VRMLNODETRANSFORM_
#define _VRMLNODETRANSFORM_

#include "VrmlNodeGroup.h"
#include "VrmlSFRotation.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeTransform : public VrmlNodeGroup
{

public:
    // Define the fields of Transform nodes
    static void initFields(VrmlNodeTransform *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeTransform(VrmlScene *);

    virtual void render(Viewer *);

    virtual void accumulateTransform(VrmlNode *);
    virtual void inverseTransform(Viewer *);
    virtual void inverseTransform(double *mat);

    //LarryD Feb 24/99
    virtual const VrmlSFVec3f &getCenter() const;
    //LarryD Feb 24/99
    virtual const VrmlSFRotation &getRotation() const;
    virtual const VrmlSFVec3f &getScale() const; //LarryD Feb 24/99
    //LarryD Feb 24/99
    virtual const VrmlSFRotation &getScaleOrientation() const;
    //LarryD Feb
    virtual const VrmlSFVec3f &getTranslation() const;

protected:
    VrmlSFVec3f d_center;
    VrmlSFRotation d_rotation;
    VrmlSFVec3f d_scale;
    VrmlSFRotation d_scaleOrientation;
    VrmlSFVec3f d_translation;

    Viewer::Object d_xformObject;
};
}
#endif //_VRMLNODETRANSFORM_
