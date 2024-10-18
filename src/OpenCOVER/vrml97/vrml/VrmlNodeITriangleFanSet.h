/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeITriangleFanSet.h

#ifndef _VRMLNODEITRIANGLEFANSET_
#define _VRMLNODEITRIANGLEFANSET_

#include "VrmlNodeIPolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeITriangleFanSet : public VrmlNodeIPolygonsCommon

{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodeITriangleFanSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeITriangleFanSet(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual VrmlNodeITriangleFanSet *toITriangleFanSet() const;
};
}
#endif // _VRMLNODEITRIANGLEFANSET_
