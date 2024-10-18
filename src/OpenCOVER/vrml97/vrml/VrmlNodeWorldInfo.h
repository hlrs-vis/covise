/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeWorldInfo.h

#ifndef _VRMLNODEWORLDINFO_
#define _VRMLNODEWORLDINFO_

#include "VrmlNode.h"
#include "VrmlMFString.h"
#include "VrmlSFString.h"
#include "VrmlSFBool.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeWorldInfo : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "WorldInfo"
    static void initFields(VrmlNodeWorldInfo *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeWorldInfo(VrmlScene *);

private:
    VrmlMFString d_info;
    VrmlSFString d_title;
    VrmlSFBool d_correctBackFaceCulling;
    VrmlSFBool d_correctSpatializedAudio;
};
}
#endif //_VRMLNODEWORLDINFO_
