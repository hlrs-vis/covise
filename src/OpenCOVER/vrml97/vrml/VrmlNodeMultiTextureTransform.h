/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMultiTextureTransform.h

#ifndef _VRMLNODEMULTITEXTURETRANSFORM_
#define _VRMLNODEMULTITEXTURETRANSFORM_

#include "VrmlNode.h"
#include "VrmlMFNode.h"

namespace vrml
{

class Viewer;

class VRMLEXPORT VrmlNodeMultiTextureTransform : public VrmlNode
{

public:
    // Define the fields of TextureTransform nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMultiTextureTransform(VrmlScene *);
    virtual ~VrmlNodeMultiTextureTransform();

    virtual VrmlNode *cloneMe() const;

    virtual void cloneChildren(VrmlNamespace *);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual VrmlNodeMultiTextureTransform *toMultiTextureTransform() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *, int numberTexture);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFNode d_textureTransform;
};
}
#endif //_VRMLNODEMULTITEXTURETRANSFORM_
