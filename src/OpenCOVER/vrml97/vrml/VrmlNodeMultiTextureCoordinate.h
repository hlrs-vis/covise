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
    static void initFields(VrmlNodeMultiTextureCoordinate *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeMultiTextureCoordinate(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *);

    virtual void copyRoutes(VrmlNamespace *ns);

    VrmlMFNode &texCoord()
    {
        return d_texCoord;
    }

private:
    VrmlMFNode d_texCoord;
};
}
#endif //_VRMLNODEMULTITEXTURECOORDINATE_
