/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeITriangleSet.h

#ifndef _VRMLNODEITRIANGLESET_
#define _VRMLNODEITRIANGLESET_

#include "VrmlNodeIPolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeITriangleSet : public VrmlNodeIPolygonsCommon

{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeITriangleSet(VrmlScene *);
    virtual ~VrmlNodeITriangleSet();

    virtual VrmlNode *cloneMe() const;

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual VrmlNodeITriangleSet *toITriangleSet() const;
};
}
#endif // _VRMLNODEITRIANGLESET_
