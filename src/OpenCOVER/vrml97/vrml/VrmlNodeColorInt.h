/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColorInt.h

#ifndef _VRMLNODECOLORINT_
#define _VRMLNODECOLORINT_

#include "VrmlNode.h"

#include "VrmlSFFloat.h"
#include "VrmlMFFloat.h"
#include "VrmlSFColor.h"
#include "VrmlMFColor.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeColorInt : public VrmlNodeChild
{

public:
    // Define the fields of ColorInt nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeColorInt(VrmlScene *scene = 0);
    virtual ~VrmlNodeColorInt();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

private:
    // Fields
    VrmlMFFloat d_key;
    VrmlMFColor d_keyValue;

    // State
    VrmlSFColor d_value;
};
}
#endif //_VRMLNODECOLORINT_
