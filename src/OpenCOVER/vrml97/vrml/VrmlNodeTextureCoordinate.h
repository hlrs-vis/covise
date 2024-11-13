/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureCoordinate.h

#ifndef _VRMLNODETEXTURECOORDINATE_
#define _VRMLNODETEXTURECOORDINATE_

#include "VrmlNode.h"
#include "VrmlMFVec2f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeTextureCoordinate : public VrmlNode
{

public:
    // Define the fields of TextureCoordinate nodes
    static void initFields(VrmlNodeTextureCoordinate *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeTextureCoordinate(VrmlScene *);

    VrmlMFVec2f &coordinate()
    {
        return d_point;
    }

private:
    VrmlMFVec2f d_point;
};
}
#endif //_VRMLNODETEXTURECOORDINATE_
