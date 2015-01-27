/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeWorldInfo.h

#ifndef _VRMLNODEWORLDINFO_
#define _VRMLNODEWORLDINFO_

#include "VrmlNode.h"
#include "VrmlMFString.h"
#include "VrmlSFString.h"
#include "VrmlSFBool.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeWorldInfo : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "WorldInfo"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeWorldInfo(VrmlScene *);
    virtual ~VrmlNodeWorldInfo();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFString d_info;
    VrmlSFString d_title;
    VrmlSFBool d_correctBackFaceCulling;
    VrmlSFBool d_correctSpatializedAudio;
};
}
#endif //_VRMLNODEWORLDINFO_
