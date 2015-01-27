/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataInteger.h

#ifndef _VRMLNODEMETADATAINTEGER_
#define _VRMLNODEMETADATAINTEGER_

#include "VrmlNodeMetadata.h"
#include "VrmlMFInt.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeMetadataInteger : public VrmlNodeMetadata

{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMetadataInteger(VrmlScene *);
    virtual ~VrmlNodeMetadataInteger();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeMetadataInteger *toMetadataInteger() const;

protected:
    VrmlMFInt d_value;
};
}
#endif // _VRMLNODEMETADATAINTEGER_
