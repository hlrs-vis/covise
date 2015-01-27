/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNormal.h

#ifndef _VRMLNODENORMAL_
#define _VRMLNODENORMAL_

#include "VrmlNode.h"
#include "VrmlMFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeNormal : public VrmlNode
{

public:
    // Define the fields of Normal nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeNormal(VrmlScene *);
    virtual ~VrmlNodeNormal();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeNormal *toNormal() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    VrmlMFVec3f &normal()
    {
        return d_vector;
    }

private:
    VrmlMFVec3f d_vector;
};
}
#endif //_VRMLNODENORMAL_
