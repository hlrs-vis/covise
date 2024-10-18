/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBox.h

#ifndef _VRMLNODEBOX_
#define _VRMLNODEBOX_

#include "VrmlNodeGeometry.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeBox : public VrmlNodeGeometry
{

public:
    // Define the fields of box nodes
    static void initFields(VrmlNodeBox *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeBox(VrmlScene *);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual VrmlNodeBox *toBox() const; //LarryD Mar 08/99
    virtual const VrmlSFVec3f &getSize() const; //LarryD Mar 08/99

protected:
    VrmlSFVec3f d_size;
};
}
#endif //_VRMLNODEBOX_
