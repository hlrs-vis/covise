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

#ifndef _VRMLNODECOLOR_
#define _VRMLNODECOLOR_

#include "VrmlNodeTemplate.h"
#include "VrmlMFColor.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeColor : public VrmlNodeTemplate
{

public:
    // Define the fields of Color nodes
    static void initFields(VrmlNodeColor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeColor(VrmlScene *);

    virtual VrmlNodeColor *toColor() const;

    VrmlMFColor &color()
    {
        return d_color;
    }

private:
    VrmlMFColor d_color;
};
}
#endif //_VRMLNODECOLOR_
