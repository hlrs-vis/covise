/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBooleanSequencer.h

#ifndef _VRMLNODEBOOLEANSEQUENCER_
#define _VRMLNODEBOOLEANSEQUENCER_

#include "VrmlNode.h"

#include "VrmlSFFloat.h"
#include "VrmlMFFloat.h"
#include "VrmlSFBool.h"
#include "VrmlMFBool.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeBooleanSequencer : public VrmlNodeChild
{

public:
    // Define the fields of BooleanSequencer nodes
    static void initFields(VrmlNodeBooleanSequencer *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeBooleanSequencer(VrmlScene *scene = 0);
    
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);


    virtual const VrmlMFFloat &getKey() const;
    virtual const VrmlMFBool &getKeyValue() const;

private:
    // Fields
    VrmlMFFloat d_key;
    VrmlMFBool d_keyValue;

    // State
    VrmlSFBool d_value;
};
}
#endif //_VRMLNODEBOOLEANSEQUENCER_
