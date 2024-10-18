/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIPolygonsCommon.h

#ifndef _VRMLNODEIPOLYGONSCOMMON_
#define _VRMLNODEIPOLYGONSCOMMON_

#include "VrmlNodePolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeIPolygonsCommon : public VrmlNodePolygonsCommon
{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodeIPolygonsCommon *node, VrmlNodeType *t);

    VrmlNodeIPolygonsCommon(VrmlScene *, const std::string &name);

protected:
    VrmlMFInt d_index;
};
}
#endif // _VRMLNODEIPOLYGONSCOMMON_
