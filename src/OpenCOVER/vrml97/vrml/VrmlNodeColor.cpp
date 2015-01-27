/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColor.cpp

#include "VrmlNodeColor.h"

#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeColor(s); }

// Define the built in VrmlNodeType:: "Color" fields

VrmlNodeType *VrmlNodeColor::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define type once.
        t = st = new VrmlNodeType("Color", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("color", VrmlField::MFCOLOR);

    return t;
}

VrmlNodeType *VrmlNodeColor::nodeType() const { return defineType(0); }

VrmlNodeColor::VrmlNodeColor(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeColor::~VrmlNodeColor()
{
}

VrmlNode *VrmlNodeColor::cloneMe() const
{
    return new VrmlNodeColor(*this);
}

VrmlNodeColor *VrmlNodeColor::toColor() const
{
    return (VrmlNodeColor *)this;
}

std::ostream &VrmlNodeColor::printFields(std::ostream &os, int indent)
{
    if (d_color.size() > 0)
        PRINT_FIELD(color);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeColor::setField(const char *fieldName,
                             const VrmlField &fieldValue)
{
    if
        TRY_FIELD(color, MFColor)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeColor::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "color") == 0)
        return &d_color;

    return VrmlNode::getField(fieldName);
}
