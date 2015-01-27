/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
// %W% %G%
//

#ifndef _VRMLNODEAPPEARANCE_
#define _VRMLNODEAPPEARANCE_

#include "VrmlNode.h"
#include "VrmlSFNode.h"
#include "VrmlSFString.h"
#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeAppearance : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "Appearance"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeAppearance(VrmlScene *);
    virtual ~VrmlNodeAppearance();

    // Copy the node.
    virtual VrmlNode *cloneMe() const;
    virtual void cloneChildren(VrmlNamespace *);

    virtual VrmlNodeAppearance *toAppearance() const;

    virtual bool isModified() const;
    virtual void clearFlags(); // Clear childrens flags too.

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName,
                          const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    VrmlNode *material()
    {
        return (VrmlNode *)d_material.get();
    }
    VrmlNode *texture()
    {
        return (VrmlNode *)d_texture.get();
    }
    VrmlNode *textureTransform()
    {
        return (VrmlNode *)d_textureTransform.get();
    }

    // additional methods for multi-texturing
    VrmlNode *texture2()
    {
        return (VrmlNode *)d_texture2.get();
    }
    VrmlNode *textureTransform2()
    {
        return (VrmlNode *)d_textureTransform2.get();
    }
    VrmlNode *texture3()
    {
        return (VrmlNode *)d_texture3.get();
    }
    VrmlNode *textureTransform3()
    {
        return (VrmlNode *)d_textureTransform3.get();
    }
    VrmlNode *texture4()
    {
        return (VrmlNode *)d_texture4.get();
    }
    VrmlNode *textureTransform4()
    {
        return (VrmlNode *)d_textureTransform4.get();
    }
    VrmlNode *texture5()
    {
        return (VrmlNode *)d_texture5.get();
    }
    VrmlNode *textureTransform5()
    {
        return (VrmlNode *)d_textureTransform5.get();
    }
    VrmlNode *texture6()
    {
        return (VrmlNode *)d_texture6.get();
    }
    VrmlNode *textureTransform6()
    {
        return (VrmlNode *)d_textureTransform6.get();
    }
    VrmlNode *texture7()
    {
        return (VrmlNode *)d_texture7.get();
    }
    VrmlNode *textureTransform7()
    {
        return (VrmlNode *)d_textureTransform7.get();
    }
    VrmlNode *texture8()
    {
        return (VrmlNode *)d_texture8.get();
    }
    VrmlNode *textureTransform8()
    {
        return (VrmlNode *)d_textureTransform8.get();
    }
    VrmlNode *texture9()
    {
        return (VrmlNode *)d_texture9.get();
    }
    VrmlNode *textureTransform9()
    {
        return (VrmlNode *)d_textureTransform9.get();
    }
    VrmlNode *texture10()
    {
        return (VrmlNode *)d_texture10.get();
    }
    VrmlNode *textureTransform10()
    {
        return (VrmlNode *)d_textureTransform10.get();
    }

protected:
    VrmlSFNode d_material;
    VrmlSFNode d_texture;
    VrmlSFNode d_textureTransform;

    // additional fields for multi-texturing
    VrmlSFNode d_texture2;
    VrmlSFNode d_textureTransform2;
    VrmlSFNode d_texture3;
    VrmlSFNode d_textureTransform3;
    VrmlSFNode d_texture4;
    VrmlSFNode d_textureTransform4;
    VrmlSFNode d_texture5;
    VrmlSFNode d_textureTransform5;
    VrmlSFNode d_texture6;
    VrmlSFNode d_textureTransform6;
    VrmlSFNode d_texture7;
    VrmlSFNode d_textureTransform7;
    VrmlSFNode d_texture8;
    VrmlSFNode d_textureTransform8;
    VrmlSFNode d_texture9;
    VrmlSFNode d_textureTransform9;
    VrmlSFNode d_texture10;
    VrmlSFNode d_textureTransform10;
    VrmlSFString d_relativeUrl;
};
}
#endif //_VRMLNODEAPPEARANCE_
