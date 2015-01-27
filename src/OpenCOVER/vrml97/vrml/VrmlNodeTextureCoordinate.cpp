/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureCoordinate.cpp

#include "VrmlNodeTextureCoordinate.h"
#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeTextureCoordinate(s);
}

// Define the built in VrmlNodeType:: "TextureCoordinate" fields

VrmlNodeType *VrmlNodeTextureCoordinate::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TextureCoordinate", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("point", VrmlField::MFVEC2F);

    return t;
}

VrmlNodeType *VrmlNodeTextureCoordinate::nodeType() const
{
    return defineType(0);
}

VrmlNodeTextureCoordinate::VrmlNodeTextureCoordinate(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeTextureCoordinate::~VrmlNodeTextureCoordinate()
{
}

VrmlNode *VrmlNodeTextureCoordinate::cloneMe() const
{
    return new VrmlNodeTextureCoordinate(*this);
}

VrmlNodeTextureCoordinate *VrmlNodeTextureCoordinate::toTextureCoordinate() const
{
    return (VrmlNodeTextureCoordinate *)this;
}

std::ostream &VrmlNodeTextureCoordinate::printFields(std::ostream &os, int indent)
{
    if (d_point.size() > 0)
        PRINT_FIELD(point);
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeTextureCoordinate::setField(const char *fieldName,
                                         const VrmlField &fieldValue)
{
    if
        TRY_FIELD(point, MFVec2f)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTextureCoordinate::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "point") == 0)
        return &d_point;

    return VrmlNode::getField(fieldName);
}
