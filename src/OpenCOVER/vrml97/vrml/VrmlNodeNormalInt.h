/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNormalInt.h

#ifndef _VRMLNODENORMALINT_
#define _VRMLNODENORMALINT_

#include "VrmlNode.h"

#include "VrmlSFFloat.h"
#include "VrmlMFFloat.h"
#include "VrmlMFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeNormalInt : public VrmlNodeChild
{

public:
    static void initFields(VrmlNodeNormalInt *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeNormalInt(VrmlScene *scene = 0);

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
#endif //_VRMLNODENORMALINT_
