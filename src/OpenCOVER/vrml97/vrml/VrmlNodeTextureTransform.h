/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureTransform.h

#ifndef _VRMLNODETEXTURETRANSFORM_
#define _VRMLNODETEXTURETRANSFORM_

#include "VrmlNode.h"
#include "VrmlSFFloat.h"
#include "VrmlSFVec2f.h"

namespace vrml
{

class Viewer;

class VRMLEXPORT VrmlNodeTextureTransform : public VrmlNode
{

public:
    // Define the fields of TextureTransform nodes
    static void initFields(VrmlNodeTextureTransform *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeTextureTransform(VrmlScene *);

    virtual void render(Viewer *);


    float *center()
    {
        return d_center.get();
    }
    float rotation()
    {
        return d_rotation.get();
    }
    float *scale()
    {
        return d_scale.get();
    }
    float *translation()
    {
        return d_translation.get();
    }

private:
    VrmlSFVec2f d_center;
    VrmlSFFloat d_rotation;
    VrmlSFVec2f d_scale;
    VrmlSFVec2f d_translation;
};
}
#endif //_VRMLNODETEXTURETRANSFORM_
