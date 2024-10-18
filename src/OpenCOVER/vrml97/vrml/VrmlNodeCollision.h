/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCollision.h

#ifndef _VRMLNODECOLLISION_
#define _VRMLNODECOLLISION_

#include "VrmlNodeGroup.h"
#include "VrmlSFBool.h"
#include "VrmlSFNode.h"
#include "VrmlSFTime.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeCollision : public VrmlNodeGroup
{

public:

    static void initFields(VrmlNodeCollision *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCollision(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;
    virtual void clearFlags(); // Clear childrens flags too.

    void render(Viewer *viewer);

    virtual void addToScene(VrmlScene *s, const char *rel);

    virtual void copyRoutes(VrmlNamespace *ns);
private:
    VrmlSFBool d_collide;
    VrmlSFNode d_proxy;

    // eventOut
    VrmlSFTime d_collideTime;
};
}
#endif //_VRMLNODECOLLISION_
