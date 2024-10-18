/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIFaceSet.h

#ifndef _VRMLNODEIFACESET_
#define _VRMLNODEIFACESET_

#include "VrmlNodeIndexedSet.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFNode.h"
#include "VrmlMFInt.h"


#include <array>
namespace vrml
{

class VRMLEXPORT VrmlNodeIFaceSet : public VrmlNodeIndexedSet
{

public:
    // Define the fields of indexed face set nodes
    static void initFields(VrmlNodeIFaceSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeIFaceSet(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual Viewer::Object insertGeometry(Viewer *v);


    virtual VrmlNodeIFaceSet *toIFaceSet() const;

    virtual VrmlNode *getNormal();
    virtual const VrmlMFInt &getNormalIndex() const;

    virtual VrmlNode *getTexCoord();
    virtual const VrmlMFInt &getTexCoordIndex() const;

    // additional fields for multi-texturing
    virtual VrmlNode *getTexCoord2();
    virtual const VrmlMFInt &getTexCoordIndex2() const;
    virtual VrmlNode *getTexCoord3();
    virtual const VrmlMFInt &getTexCoordIndex3() const;
    virtual VrmlNode *getTexCoord4();
    virtual const VrmlMFInt &getTexCoordIndex4() const;
    virtual VrmlNode *getTexCoord5();
    virtual const VrmlMFInt &getTexCoordIndex5() const;
    virtual VrmlNode *getTexCoord6();
    virtual const VrmlMFInt &getTexCoordIndex6() const;
    virtual VrmlNode *getTexCoord7();
    virtual const VrmlMFInt &getTexCoordIndex7() const;
    virtual VrmlNode *getTexCoord8();
    virtual const VrmlMFInt &getTexCoordIndex8() const;
    virtual VrmlNode *getTexCoord9();
    virtual const VrmlMFInt &getTexCoordIndex9() const;
    virtual VrmlNode *getTexCoord10();
    virtual const VrmlMFInt &getTexCoordIndex10() const;

    virtual bool getCcw() // LarryD  Feb18/99
    {
        return d_ccw.get();
    }
    virtual bool getConvex() // LarryD Feb18/99
    {
        return d_convex.get();
    }
    virtual float getCreaseAngle() // LarryD  Feb18/99
    {
        return d_creaseAngle.get();
    }
    virtual bool getNormalPerVertex() // LarryD  Feb18/99
    {
        return d_normalPerVertex.get();
    }
    virtual bool getSolid() // LarryD  Feb18/99
    {
        return d_solid.get();
    }

protected:
    VrmlSFBool d_ccw;
    VrmlSFBool d_convex;
    VrmlSFFloat d_creaseAngle;
    VrmlSFNode d_normal;
    VrmlMFInt d_normalIndex;
    VrmlSFBool d_normalPerVertex;
    VrmlSFBool d_solid;

    // additional fields for multi-texturing
    static const size_t MAX_TEXCOORDS = 10;
    std::array<VrmlSFNode, MAX_TEXCOORDS> d_texCoords;
    std::array<VrmlMFInt, MAX_TEXCOORDS> d_texCoordIndices;

};
}
#endif // _VRMLNODEIFACESET_
