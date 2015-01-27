/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureCoordinateGenerator.cpp

#include "VrmlNodeTextureCoordinateGenerator.h"
#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeTextureCoordinateGenerator(s);
}

// Define the built in VrmlNodeType:: "TextureCoordinateGenerator" fields

VrmlNodeType *VrmlNodeTextureCoordinateGenerator::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TextureCoordinateGenerator", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("mode", VrmlField::SFSTRING);
    t->addExposedField("parameter", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeTextureCoordinateGenerator::nodeType() const
{
    return defineType(0);
}

VrmlNodeTextureCoordinateGenerator::VrmlNodeTextureCoordinateGenerator(VrmlScene *scene)
    : VrmlNode(scene)
    , d_mode("SPHERE")
{
}

VrmlNodeTextureCoordinateGenerator::~VrmlNodeTextureCoordinateGenerator()
{
}

VrmlNode *VrmlNodeTextureCoordinateGenerator::cloneMe() const
{
    return new VrmlNodeTextureCoordinateGenerator(*this);
}

VrmlNodeTextureCoordinateGenerator *VrmlNodeTextureCoordinateGenerator::toTextureCoordinateGenerator() const
{
    return (VrmlNodeTextureCoordinateGenerator *)this;
}

std::ostream &VrmlNodeTextureCoordinateGenerator::printFields(std::ostream &os, int indent)
{
    if (d_mode.get() && strcmp(d_mode.get(), "SPHERE"))
        PRINT_FIELD(mode);
    if (d_parameter.size() > 0)
        PRINT_FIELD(parameter);
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeTextureCoordinateGenerator::setField(const char *fieldName,
                                                  const VrmlField &fieldValue)
{
    if
        TRY_FIELD(mode, SFString)
    else if
        TRY_FIELD(parameter, MFFloat)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTextureCoordinateGenerator::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "mode") == 0)
        return &d_mode;

    if (strcmp(fieldName, "parameter") == 0)
        return &d_parameter;

    return VrmlNode::getField(fieldName);
}
