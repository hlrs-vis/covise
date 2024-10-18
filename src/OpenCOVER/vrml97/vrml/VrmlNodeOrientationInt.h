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
    static void initFields(VrmlNodeOrientationInt *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeOrientationInt(VrmlScene *scene = 0);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

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
