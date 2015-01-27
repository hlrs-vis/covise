/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataFloat.h

#ifndef _VRMLNODEMETADATAFLOAT_
#define _VRMLNODEMETADATAFLOAT_

#include "VrmlNodeMetadata.h"
#include "VrmlMFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeMetadataFloat : public VrmlNodeMetadata

{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMetadataFloat(VrmlScene *);
    virtual ~VrmlNodeMetadataFloat();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeMetadataFloat *toMetadataFloat() const;

protected:
    VrmlMFFloat d_value;
};
}
#endif // _VRMLNODEMETADATAFLOAT_
