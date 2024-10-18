/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTriangleFanSet.h

#ifndef _VRMLNODETRIANGLEFANSET_
#define _VRMLNODETRIANGLEFANSET_

#include "VrmlNodePolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeTriangleFanSet : public VrmlNodePolygonsCommon

{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodeTriangleFanSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeTriangleFanSet(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual VrmlNodeTriangleFanSet *toTriangleFanSet() const;

protected:
    VrmlMFInt d_fanCount;
};
}
#endif // _VRMLNODETRIANGLEFANSET_
