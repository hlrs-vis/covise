/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCoordinateInt.h

#ifndef _VRMLNODECOORDINATEINT_
#define _VRMLNODECOORDINATEINT_

#include "VrmlNode.h"

#include "VrmlSFFloat.h"
#include "VrmlMFFloat.h"
#include "VrmlMFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeCoordinateInt : public VrmlNodeChild
{

public:
    // Define the fields of CoordinateInt nodes
    static void initFields(VrmlNodeCoordinateInt *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCoordinateInt(VrmlScene *scene = 0);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

private:
    // Fields
    VrmlMFFloat d_key;
    VrmlMFVec3f d_keyValue;

    // State
    VrmlMFVec3f d_value;
};
}
#endif //_VRMLNODECOORDINATEINT_
