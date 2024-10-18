/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCylinder.h

#ifndef _VRMLNODECYLINDER_
#define _VRMLNODECYLINDER_

#include "VrmlNodeGeometry.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeCylinder : public VrmlNodeGeometry
{

public:
    // Define the fields of cylinder nodes
    static void initFields(VrmlNodeCylinder *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCylinder(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *);

    //LarryD Mar 08/99
    virtual VrmlNodeCylinder *toCylinder() const;
    virtual bool getBottom() //LarryD Mar 08/99
    {
        return d_bottom.get();
    }
    virtual bool getSide() //LarryD Mar 08/99
    {
        return d_side.get();
    }
    virtual bool getTop() //LarryD Mar 08/99
    {
        return d_top.get();
    }
    virtual float getHeight() //LarryD Mar 08/99
    {
        return d_height.get();
    }
    virtual float getRadius() //LarryD Mar 08/99
    {
        return d_radius.get();
    }

protected:
    VrmlSFBool d_bottom;
    VrmlSFFloat d_height;
    VrmlSFFloat d_radius;
    VrmlSFBool d_side;
    VrmlSFBool d_top;
};
}
#endif //_VRMLNODECYLINDER_
