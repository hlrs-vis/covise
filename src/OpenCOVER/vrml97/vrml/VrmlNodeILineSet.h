/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeILineSet.h

#ifndef _VRMLNODEILINESET_
#define _VRMLNODEILINESET_

#include "VrmlNodeIndexedSet.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlMFInt.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeILineSet : public VrmlNodeIndexedSet
{

public:
    // Define the fields of indexed line set nodes
    static void initFields(VrmlNodeILineSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeILineSet(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *);

    virtual Viewer::Object insertGeometry(Viewer *v);

protected:
    // All fields are defined in VrmlNodeIndexedSet
};
}
#endif // _VRMLNODEILINESET_
