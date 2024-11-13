/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSphere.h

#ifndef _VRMLNODESPHERE_
#define _VRMLNODESPHERE_

#include "VrmlNodeGeometry.h"
#include "VrmlSFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeSphere : public VrmlNodeGeometry
{

public:
    // Define the fields of sphere nodes
    static void initFields(VrmlNodeSphere *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeSphere(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual float getRadius() //LarryD Mar 08/99
    {
        return d_radius.get();
    }

protected:
    VrmlSFFloat d_radius;
};
}
#endif //_VRMLNODESPHERE_
