/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeDirLight.h

#ifndef _VRMLNODEDIRLIGHT_
#define _VRMLNODEDIRLIGHT_

#include "VrmlNodeLight.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeDirLight : public VrmlNodeLight
{

public:
    static void initFields(VrmlNodeDirLight *node, VrmlNodeType *t);
    static const char *name();
    
    VrmlNodeDirLight(VrmlScene *);


    virtual void render(Viewer *);

    //LarryD Mar 04/99
    virtual const VrmlSFVec3f &getDirection() const;

protected:
    VrmlSFVec3f d_direction;
};
}
#endif //_VRMLNODEDIRLIGHT_
