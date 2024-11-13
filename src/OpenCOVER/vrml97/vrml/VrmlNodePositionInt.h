/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePositionInt.h

#ifndef _VRMLNODEPOSITIONINT_
#define _VRMLNODEPOSITIONINT_

#include "VrmlNode.h"

#include "VrmlSFFloat.h"
#include "VrmlMFFloat.h"
#include "VrmlSFVec3f.h"
#include "VrmlMFVec3f.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodePositionInt : public VrmlNodeChild
{

public:
    
    static void initFields(VrmlNodePositionInt *node, VrmlNodeType *t);
    static const char *name();  

    VrmlNodePositionInt(VrmlScene *scene = 0);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual const VrmlMFFloat &getKey() const;
    virtual const VrmlMFVec3f &getKeyValue() const;

private:
    // Fields
    VrmlMFFloat d_key;
    VrmlMFVec3f d_keyValue;

    // State
    VrmlSFVec3f d_value;
};
}
#endif //_VRMLNODEPOSITIONINT_
