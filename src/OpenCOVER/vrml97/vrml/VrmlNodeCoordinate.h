/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCoordinate.h

#ifndef _VRMLNODECOORDINATE_
#define _VRMLNODECOORDINATE_

#include "VrmlNode.h"
#include "VrmlMFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeCoordinate : public VrmlNode
{

public:
    // Define the fields of Coordinate nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeCoordinate(VrmlScene *);
    virtual ~VrmlNodeCoordinate();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeCoordinate *toCoordinate() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    VrmlMFVec3f &coordinate()
    {
        return d_point;
    }

    virtual int getNumberCoordinates();

private:
    VrmlMFVec3f d_point;
};
}
#endif //_VRMLNODECOORDINATE_
