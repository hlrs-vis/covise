/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCoordinate.h

#ifndef _VRMLNODECOORDINATE_
#define _VRMLNODECOORDINATE_

#include "VrmlNode.h"
#include "VrmlMFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeCoordinate : public VrmlNode
{

public:
    // Define the fields of Coordinate nodes
    static void initFields(VrmlNodeCoordinate *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCoordinate(VrmlScene *);

    VrmlMFVec3f &coordinate()
    {
        return d_point;
    }

    virtual int getNumberCoordinates();

private:
    VrmlMFVec3f d_point;
};
}
#endif //_VRMLNODECOORDINATE_
