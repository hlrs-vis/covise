/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeITriangleStripSet.h

#ifndef _VRMLNODEITRIANGLESTRIPSET_
#define _VRMLNODEITRIANGLESTRIPSET_

#include "VrmlNodeIPolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeITriangleStripSet : public VrmlNodeIPolygonsCommon

{

public:
    // Define the fields of indexed face set nodes
  static void initFields(VrmlNodeITriangleStripSet *node, VrmlNodeType *t);
  static const char *name();

    VrmlNodeITriangleStripSet(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *v);
};
}
#endif // _VRMLNODEITRIANGLESTRIPSET_
