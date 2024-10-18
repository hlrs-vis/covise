/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMultiTextureCoordinate.cpp

#include "VrmlNodeMultiTextureCoordinate.h"
#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeMultiTextureCoordinate::initFields(VrmlNodeMultiTextureCoordinate *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t, exposedField("texCoord", node->d_texCoord));
}

const char *VrmlNodeMultiTextureCoordinate::name()
{
    return "MultiTextureCoordinate";
}

VrmlNodeMultiTextureCoordinate::VrmlNodeMultiTextureCoordinate(VrmlScene *scene)
    : VrmlNodeTemplate(scene, name())
{
}

void VrmlNodeMultiTextureCoordinate::cloneChildren(VrmlNamespace *ns)
{
    // Replace references with clones
    int n = d_texCoord.size();
    VrmlNode **kids = d_texCoord.get();
    for (int i = 0; i < n; ++i)
    {
        if (!kids[i])
            continue;
        VrmlNode *newKid = kids[i]->clone(ns)->reference();
        kids[i]->dereference();
        kids[i] = newKid;
        kids[i]->parentList.push_back(this);
    }
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeMultiTextureCoordinate::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns); // Copy my routes

    // Copy subnode routes
    int n = d_texCoord.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_texCoord[i] == NULL)
            continue;
        d_texCoord[i]->copyRoutes(ns);
    }
    nodeStack.pop_front();
}

VrmlNodeMultiTextureCoordinate *VrmlNodeMultiTextureCoordinate::toMultiTextureCoordinate() const
{
    return (VrmlNodeMultiTextureCoordinate *)this;
}
