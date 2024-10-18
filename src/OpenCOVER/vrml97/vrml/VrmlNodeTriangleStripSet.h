/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTriangleStripSet.h

#ifndef _VRMLNODETRIANGLESTRIPSET_
#define _VRMLNODETRIANGLESTRIPSET_

#include "VrmlNodePolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeTriangleStripSet : public VrmlNodePolygonsCommon
{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodeTriangleStripSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeTriangleStripSet(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual VrmlNodeTriangleStripSet *toTriangleStripSet() const;

protected:
    VrmlMFInt d_stripCount;
};
}
#endif // _VRMLNODETRIANGLESTRIPSET_
