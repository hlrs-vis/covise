/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeChild.h

#ifndef _VRMLNODECHILD_
#define _VRMLNODECHILD_

#include "VrmlNode.h"

namespace vrml
{
class VRMLEXPORT VrmlNodeScene;

class VRMLEXPORT VrmlNodeChild : public VrmlNode
{

public:
    static void initFields(VrmlNodeChild *node, VrmlNodeType *t);
    VrmlNodeChild(VrmlScene *scene, const std::string& name);

};

}
#endif //_VRMLNODECHILD_
