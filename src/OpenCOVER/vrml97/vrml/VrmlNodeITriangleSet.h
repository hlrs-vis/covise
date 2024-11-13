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
  static void initFields(VrmlNodeITriangleSet *node, VrmlNodeType *t);
  static const char *name();

    VrmlNodeITriangleSet(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *v);
};
}
#endif // _VRMLNODEITRIANGLESET_
