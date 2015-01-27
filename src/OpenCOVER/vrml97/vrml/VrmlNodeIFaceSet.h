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

namespace vrml
{

class VRMLEXPORT VrmlNodeIFaceSet : public VrmlNodeIndexedSet
{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeIFaceSet(VrmlScene *);
    virtual ~VrmlNodeIFaceSet();

    virtual VrmlNode *cloneMe() const;
    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual Viewer::Object insertGeometry(Viewer *v);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

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
    VrmlSFNode d_texCoord;
    VrmlMFInt d_texCoordIndex;

    // additional fields for multi-texturing
    VrmlSFNode d_texCoord2;
    VrmlMFInt d_texCoordIndex2;
    VrmlSFNode d_texCoord3;
    VrmlMFInt d_texCoordIndex3;
    VrmlSFNode d_texCoord4;
    VrmlMFInt d_texCoordIndex4;
    VrmlSFNode d_texCoord5;
    VrmlMFInt d_texCoordIndex5;
    VrmlSFNode d_texCoord6;
    VrmlMFInt d_texCoordIndex6;
    VrmlSFNode d_texCoord7;
    VrmlMFInt d_texCoordIndex7;
    VrmlSFNode d_texCoord8;
    VrmlMFInt d_texCoordIndex8;
    VrmlSFNode d_texCoord9;
    VrmlMFInt d_texCoordIndex9;
    VrmlSFNode d_texCoord10;
    VrmlMFInt d_texCoordIndex10;
};
}
#endif // _VRMLNODEIFACESET_
