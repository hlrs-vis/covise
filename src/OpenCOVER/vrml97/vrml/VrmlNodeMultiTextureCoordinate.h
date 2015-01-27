/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMultiTextureCoordinate.h

#ifndef _VRMLNODEMULTITEXTURECOORDINATE_
#define _VRMLNODEMULTITEXTURECOORDINATE_

#include "VrmlNode.h"
#include "VrmlMFNode.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeMultiTextureCoordinate : public VrmlNode
{

public:
    // Define the fields of TextureCoordinate nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMultiTextureCoordinate(VrmlScene *);
    virtual ~VrmlNodeMultiTextureCoordinate();

    virtual VrmlNode *cloneMe() const;

    virtual void cloneChildren(VrmlNamespace *);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual VrmlNodeMultiTextureCoordinate *toMultiTextureCoordinate() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    VrmlMFNode &texCoord()
    {
        return d_texCoord;
    }

private:
    VrmlMFNode d_texCoord;
};
}
#endif //_VRMLNODEMULTITEXTURECOORDINATE_
