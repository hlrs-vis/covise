/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMultiTextureTransform.cpp

#include "VrmlNodeMultiTextureTransform.h"
#include "VrmlNodeTextureTransform.h"
#include "VrmlNodeType.h"
#include "MathUtils.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeMultiTextureTransform::initFields(VrmlNodeMultiTextureTransform *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t, exposedField("textureTransform", node->d_textureTransform));
}

const char *VrmlNodeMultiTextureTransform::name()
{
    return "MultiTextureTransform";
}

VrmlNodeMultiTextureTransform::VrmlNodeMultiTextureTransform(VrmlScene *scene)
    : VrmlNode(scene, name())
    , d_textureTransform(0)
{
}

VrmlNodeMultiTextureTransform::~VrmlNodeMultiTextureTransform()
{
    // delete d_viewerObject...
    while (d_textureTransform.size())
    {
        if (d_textureTransform[0])
        {
            d_textureTransform.removeNode(d_textureTransform[0]);
        }
    }
}

void VrmlNodeMultiTextureTransform::render(Viewer *viewer, int numberTexture)
{
    d_textureTransform[numberTexture]->render(viewer);
    clearModified();
}

void VrmlNodeMultiTextureTransform::cloneChildren(VrmlNamespace *ns)
{
    // Replace references with clones
    int n = d_textureTransform.size();
    VrmlNode **kids = d_textureTransform.get();
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

void VrmlNodeMultiTextureTransform::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns); // Copy my routes

    // Copy subnode routes
    int n = d_textureTransform.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_textureTransform[i] == NULL)
            continue;
        d_textureTransform[i]->copyRoutes(ns);
    }
    nodeStack.pop_front();
}
