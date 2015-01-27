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
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeBooleanSequencer(VrmlScene *scene = 0);
    virtual ~VrmlNodeBooleanSequencer();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeBooleanSequencer *toBooleanSequencer() const;
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
