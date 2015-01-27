/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataString.h

#ifndef _VRMLNODEMETADATASTRING_
#define _VRMLNODEMETADATASTRING_

#include "VrmlNodeMetadata.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeMetadataString : public VrmlNodeMetadata

{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMetadataString(VrmlScene *);
    virtual ~VrmlNodeMetadataString();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeMetadataString *toMetadataString() const;

protected:
    VrmlMFString d_value;
};
}
#endif // _VRMLNODEMETADATASTRING_
