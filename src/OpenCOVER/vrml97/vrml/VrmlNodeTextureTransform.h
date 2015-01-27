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
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTextureTransform(VrmlScene *);
    virtual ~VrmlNodeTextureTransform();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeTextureTransform *toTextureTransform() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

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
