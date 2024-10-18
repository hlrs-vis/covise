/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePolygonsCommon.h

#ifndef _VRMLNODEPOLYGONSCOMMON_
#define _VRMLNODEPOLYGONSCOMMON_

#include "VrmlNodeColoredSet.h"
#include "VrmlSFBool.h"
#include "VrmlSFNode.h"
#include "VrmlMFNode.h"
#include "VrmlMFInt.h"

#define CREASE_ANGLE 1.57f

namespace vrml
{

class VRMLEXPORT VrmlNodePolygonsCommon : public VrmlNodeColoredSet
{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodePolygonsCommon *node, VrmlNodeType *t);

    VrmlNodePolygonsCommon(VrmlScene *, const std::string &name);

    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual bool getCcw() // LarryD  Feb18/99
    {
        return d_ccw.get();
    }
    virtual bool getNormalPerVertex() // LarryD  Feb18/99
    {
        return d_normalPerVertex.get();
    }
    virtual bool getSolid() // LarryD  Feb18/99
    {
        return d_solid.get();
    }

    virtual VrmlNode *getNormal();

    virtual VrmlNode *getTexCoord();

protected:
    VrmlSFNode d_fogCoord;
    VrmlMFNode d_attrib;
    VrmlSFBool d_ccw;
    VrmlSFNode d_normal;
    VrmlSFBool d_normalPerVertex;
    VrmlSFBool d_solid;
    VrmlSFNode d_texCoord;
};
}
#endif // _VRMLNODEPOLYGONSCOMMON_
