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

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeMultiTextureCoordinate(s);
}

// Define the built in VrmlNodeType:: "MultiTextureCoordinate" fields

VrmlNodeType *VrmlNodeMultiTextureCoordinate::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MultiTextureCoordinate", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("texCoord", VrmlField::MFNODE);

    return t;
}

VrmlNodeType *VrmlNodeMultiTextureCoordinate::nodeType() const
{
    return defineType(0);
}

VrmlNodeMultiTextureCoordinate::VrmlNodeMultiTextureCoordinate(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeMultiTextureCoordinate::~VrmlNodeMultiTextureCoordinate()
{
}

VrmlNode *VrmlNodeMultiTextureCoordinate::cloneMe() const
{
    return new VrmlNodeMultiTextureCoordinate(*this);
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

std::ostream &VrmlNodeMultiTextureCoordinate::printFields(std::ostream &os, int indent)
{
    if (d_texCoord.size() > 0)
        PRINT_FIELD(texCoord);
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMultiTextureCoordinate::setField(const char *fieldName,
                                              const VrmlField &fieldValue)
{
    if
        TRY_FIELD(texCoord, MFNode)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeMultiTextureCoordinate::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "texCoord") == 0)
        return &d_texCoord;

    return VrmlNode::getField(fieldName);
}
