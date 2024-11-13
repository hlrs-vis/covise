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
#include <array>
namespace vrml
{

class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeAppearance : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "Appearance"
    static void initFields(VrmlNodeAppearance *node, vrml::VrmlNodeType *t);
    static const char *name() { return "Appearance"; }

    VrmlNodeAppearance(VrmlScene *);

    // Copy the node.
    virtual void cloneChildren(VrmlNamespace *) override;

    virtual bool isModified() const override;
    virtual void clearFlags() override; // Clear childrens flags too.

    virtual void addToScene(VrmlScene *s, const char *relativeUrl) override;

    virtual void copyRoutes(VrmlNamespace *ns) override;
    virtual void render(Viewer *) override;

    VrmlNode *material()
    {
        return (VrmlNode *)d_material.get();
    }
    VrmlNode *texture(int i = 0)
    {
        return (VrmlNode *)d_textures[i].get();
    }
    VrmlNode *textureTransform(int i = 0)
    {
        return (VrmlNode *)d_textureTransforms[i].get();
    }

    bool isOnlyGeometry() const override;

protected:
    VrmlSFNode d_material;
    // additional fields for multi-texturing
    static const int MAX_TEXTURES = 10;
    std::array<VrmlSFNode, MAX_TEXTURES> d_textures;
    std::array<VrmlSFNode, MAX_TEXTURES> d_textureTransforms;

    VrmlSFString d_relativeUrl;
};
}
#endif //_VRMLNODEAPPEARANCE_
