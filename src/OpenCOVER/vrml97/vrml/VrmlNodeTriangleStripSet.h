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
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTriangleStripSet(VrmlScene *);
    virtual ~VrmlNodeTriangleStripSet();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeTriangleStripSet *toTriangleStripSet() const;

protected:
    VrmlMFInt d_stripCount;
};
}
#endif // _VRMLNODETRIANGLESTRIPSET_
