/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataFloat.cpp

#include "VrmlNodeMetadataFloat.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeMetadataFloat(s); }

// Define the built in VrmlNodeType:: "MetadataFloat" fields

VrmlNodeType *VrmlNodeMetadataFloat::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MetadataFloat", creator);
    }

    VrmlNodeMetadata::defineType(t); // Parent class

    t->addExposedField("value", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeMetadataFloat::nodeType() const { return defineType(0); }

VrmlNodeMetadataFloat::VrmlNodeMetadataFloat(VrmlScene *scene)
    : VrmlNodeMetadata(scene)
{
}

VrmlNodeMetadataFloat::~VrmlNodeMetadataFloat()
{
}

VrmlNode *VrmlNodeMetadataFloat::cloneMe() const
{
    return new VrmlNodeMetadataFloat(*this);
}

std::ostream &VrmlNodeMetadataFloat::printFields(std::ostream &os, int indent)
{
    if (!d_value.get())
        PRINT_FIELD(value);

    VrmlNodeMetadata::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadataFloat::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_FIELD(value, MFFloat)
    else
        VrmlNodeMetadata::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadataFloat::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "value") == 0)
        return &d_value;
    return VrmlNodeMetadata::getField(fieldName);
}

VrmlNodeMetadataFloat *VrmlNodeMetadataFloat::toMetadataFloat() const
{
    return (VrmlNodeMetadataFloat *)this;
}
