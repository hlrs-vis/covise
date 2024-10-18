/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCone.h

#ifndef _VRMLNODECONE_
#define _VRMLNODECONE_

#include "VrmlNodeGeometry.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeCone : public VrmlNodeGeometry
{

public:
    // Define the fields of cone nodes
    static void initFields(VrmlNodeCone *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCone(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual VrmlNodeCone *toCone() const; //LarryD Mar 08/99
    virtual bool getBottom() //LarryD Mar 08/99
    {
        return d_bottom.get();
    }
    virtual bool getSide() //LarryD Mar 08/99
    {
        return d_side.get();
    }
    virtual float getBottomRadius() //LarryD Mar 08/99
    {
        return d_bottomRadius.get();
    }
    virtual float getHeight() //LarryD Mar 08/99
    {
        return d_height.get();
    }

protected:
    VrmlSFBool d_bottom;
    VrmlSFFloat d_bottomRadius;
    VrmlSFFloat d_height;
    VrmlSFBool d_side;
};
}
#endif //_VRMLNODECONE_
