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

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeMultiTextureTransform(s);
}

// Define the built in VrmlNodeType:: "MultiTextureTransform" fields

VrmlNodeType *VrmlNodeMultiTextureTransform::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("MultiTextureTransform", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("textureTransform", VrmlField::MFNODE);

    return t;
}

VrmlNodeType *VrmlNodeMultiTextureTransform::nodeType() const
{
    return defineType(0);
}

VrmlNodeMultiTextureTransform::VrmlNodeMultiTextureTransform(VrmlScene *scene)
    : VrmlNode(scene)
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

VrmlNode *VrmlNodeMultiTextureTransform::cloneMe() const
{
    return new VrmlNodeMultiTextureTransform(*this);
}

VrmlNodeMultiTextureTransform *VrmlNodeMultiTextureTransform::toMultiTextureTransform() const
{
    return (VrmlNodeMultiTextureTransform *)this;
}

std::ostream &VrmlNodeMultiTextureTransform::printFields(std::ostream &os, int indent)
{
    if (d_textureTransform.size() > 0)
        PRINT_FIELD(textureTransform);

    return os;
}

void VrmlNodeMultiTextureTransform::render(Viewer *viewer, int numberTexture)
{
    d_textureTransform[numberTexture]->render(viewer);
    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeMultiTextureTransform::setField(const char *fieldName,
                                             const VrmlField &fieldValue)
{
    if
        TRY_FIELD(textureTransform, MFNode)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeMultiTextureTransform::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "textureTransform") == 0)
        return &d_textureTransform;

    return VrmlNode::getField(fieldName);
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
