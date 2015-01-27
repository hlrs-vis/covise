/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeOrientationInt.h

#ifndef _VRMLNODEORIENTATIONINT_
#define _VRMLNODEORIENTATIONINT_

#include "VrmlNode.h"

#include "VrmlSFFloat.h"
#include "VrmlMFFloat.h"
#include "VrmlSFRotation.h"
#include "VrmlMFRotation.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeOrientationInt : public VrmlNodeChild
{

public:
    // Define the fields of OrientationInt nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeOrientationInt(VrmlScene *scene = 0);
    virtual ~VrmlNodeOrientationInt();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeOrientationInt *toOrientationInt() const;
    virtual const VrmlMFFloat &getKey() const;
    virtual const VrmlMFRotation &getKeyValue() const;

private:
    // Fields
    VrmlMFFloat d_key;
    VrmlMFRotation d_keyValue;

    // State
    VrmlSFRotation d_value;
};
}
#endif //_VRMLNODEORIENTATIONINT_
