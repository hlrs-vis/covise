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
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);

    VrmlNodeIPolygonsCommon(VrmlScene *);
    virtual ~VrmlNodeIPolygonsCommon();

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

protected:
    VrmlMFInt d_index;
};
}
#endif // _VRMLNODEIPOLYGONSCOMMON_
