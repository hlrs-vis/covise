/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColor.h

#ifndef _VRMLNODECOLORRGBA_
#define _VRMLNODECOLORRGBA_

#include "VrmlNode.h"
#include "VrmlMFColorRGBA.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeColorRGBA : public VrmlNode
{

public:
    // Define the fields of Color nodes
    static void initFields(VrmlNodeColorRGBA *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeColorRGBA(VrmlScene *);

    VrmlMFColorRGBA &color()
    {
        return d_color;
    }

private:
    VrmlMFColorRGBA d_color;
};
}
#endif //_VRMLNODECOLORRGBA_
