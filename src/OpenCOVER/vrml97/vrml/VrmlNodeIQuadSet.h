/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIQuadSet.h

#ifndef _VRMLNODEIQUADSET_
#define _VRMLNODEIQUADSET_

#include "VrmlNodeIPolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeIQuadSet : public VrmlNodeIPolygonsCommon

{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeIQuadSet(VrmlScene *);
    virtual ~VrmlNodeIQuadSet();

    virtual VrmlNode *cloneMe() const;

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual VrmlNodeIQuadSet *toIQuadSet() const;
};
}
#endif // _VRMLNODEIQUADSET_
