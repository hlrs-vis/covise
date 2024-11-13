/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeFontStyle.h

#ifndef _VRMLNODEFONTSTYLE_
#define _VRMLNODEFONTSTYLE_

#include "VrmlNode.h"
#include "VrmlMFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFString.h"
#include "VrmlSFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeFontStyle : public VrmlNode
{

public:
    // Define the fields of FontStyle nodes
    static void initFields(VrmlNodeFontStyle *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeFontStyle(VrmlScene *);

    VrmlMFString &justify()
    {
        return d_justify;
    }
    float size()
    {
        return d_size.get();
    }

private:
    VrmlMFString d_family;
    VrmlSFBool d_horizontal;
    VrmlMFString d_justify;
    VrmlSFString d_language;
    VrmlSFBool d_leftToRight;
    VrmlSFFloat d_size;
    VrmlSFFloat d_spacing;
    VrmlSFString d_style;
    VrmlSFBool d_topToBottom;
};
}
#endif //_VRMLNODEFONTSTYLE_
