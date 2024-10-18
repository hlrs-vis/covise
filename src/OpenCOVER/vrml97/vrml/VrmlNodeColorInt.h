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
    static void initFields(VrmlNodeColorInt *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeColorInt(VrmlScene *scene = 0);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

private:
    // Fields
    VrmlMFFloat d_key;
    VrmlMFColor d_keyValue;

    // State
    VrmlSFColor d_value;
};
}
#endif //_VRMLNODECOLORINT_
