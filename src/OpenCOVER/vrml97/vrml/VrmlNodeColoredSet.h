/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColoredSet.h

#ifndef _VRMLNODECOLOREDSET_
#define _VRMLNODECOLOREDSET_

#include "VrmlNodeGeometry.h"

#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFNode.h"
#include "VrmlMFInt.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeColoredSet : public VrmlNodeGeometry
{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodeColoredSet *node, VrmlNodeType *t);

    VrmlNodeColoredSet(VrmlScene *, const std::string &name);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual VrmlNodeColor *color();

    virtual VrmlNode *getCoordinate();
    virtual bool getColorPerVertex() // LarryD  Feb18/99
    {
        return d_colorPerVertex.get();
    }

    Viewer::Object insertGeometry(Viewer *viewer, unsigned int optMask,
                                  VrmlMFInt &coordIndex,
                                  VrmlMFInt &colorIndex,
                                  VrmlSFFloat &creaseAngle,
                                  VrmlSFNode &normal,
                                  VrmlMFInt &normalIndex,
                                  VrmlSFNode &texCoord,
                                  VrmlMFInt &texCoordIndex,
                                  VrmlSFNode &texCoord2,
                                  VrmlMFInt &texCoordIndex2,
                                  VrmlSFNode &texCoord3,
                                  VrmlMFInt &texCoordIndex3,
                                  VrmlSFNode &texCoord4,
                                  VrmlMFInt &texCoordIndex4);

    Viewer::Object insertGeometry(Viewer *viewer, unsigned int optMask,
                                  VrmlMFInt &coordIndex,
                                  VrmlMFInt &colorIndex,
                                  VrmlSFFloat &creaseAngle,
                                  VrmlSFNode &normal,
                                  VrmlMFInt &normalIndex,
                                  VrmlSFNode &texCoord,
                                  VrmlMFInt &texCoordIndex);

protected:
    VrmlSFNode d_color;
    VrmlSFBool d_colorPerVertex;

    VrmlSFNode d_coord;
};
}
#endif //_VRMLNODECOLOREDSET_
