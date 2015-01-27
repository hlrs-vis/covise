/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColorRGBA.cpp

#include "VrmlNodeColorRGBA.h"

#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeColorRGBA(s); }

// Define the built in VrmlNodeType:: "ColorRGBA" fields

VrmlNodeType *VrmlNodeColorRGBA::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define type once.
        t = st = new VrmlNodeType("ColorRGBA", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("color", VrmlField::MFCOLORRGBA);

    return t;
}

VrmlNodeType *VrmlNodeColorRGBA::nodeType() const { return defineType(0); }

VrmlNodeColorRGBA::VrmlNodeColorRGBA(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeColorRGBA::~VrmlNodeColorRGBA()
{
}

VrmlNode *VrmlNodeColorRGBA::cloneMe() const
{
    return new VrmlNodeColorRGBA(*this);
}

VrmlNodeColorRGBA *VrmlNodeColorRGBA::toColorRGBA() const
{
    return (VrmlNodeColorRGBA *)this;
}

std::ostream &VrmlNodeColorRGBA::printFields(std::ostream &os, int indent)
{
    if (d_color.size() > 0)
        PRINT_FIELD(color);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeColorRGBA::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(color, MFColorRGBA)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeColorRGBA::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "color") == 0)
        return &d_color;

    return VrmlNode::getField(fieldName);
}
