/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBindable.h

#ifndef _VRMLNODEBINDABLE_
#define _VRMLNODEBINDABLE_

#include "VrmlNode.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeType;
class VRMLEXPORT VrmlField;
class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeBindable : public VrmlNodeChild
{

public:
    VrmlNodeBindable::VrmlNodeBindable(VrmlScene *s = 0)
        : VrmlNodeChild(s)
    {
    }

    // Define the fields of all built in bindable nodes
    static VrmlNodeType *defineType(VrmlNodeType *t)
    {
        return VrmlNode::defineType(t);
    }

    virtual bool isBindableNode()
    {
        return true;
    }
};
}
#endif //_VRMLNODEBINDABLE_
