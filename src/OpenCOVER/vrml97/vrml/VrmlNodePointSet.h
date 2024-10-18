/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePointSet.h

#ifndef _VRMLNODEPOINTSET_
#define _VRMLNODEPOINTSET_

#include "VrmlNodeGeometry.h"
#include "VrmlSFNode.h"

namespace vrml
{

class VRMLEXPORT VrmlNodePointSet : public VrmlNodeGeometry
{

public:
    // Define the fields of pointSet nodes
    static void initFields(VrmlNodePointSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodePointSet(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);


    virtual Viewer::Object insertGeometry(Viewer *);

protected:
    VrmlSFNode d_color;
    VrmlSFNode d_coord;
};
}
#endif //_VRMLNODEPOINTSET_
