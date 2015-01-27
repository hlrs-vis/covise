/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataString.cpp

#include "VrmlNodeMetadataString.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeMetadataString(s); }

// Define the built in VrmlNodeType:: "MetadataString" fields

VrmlNodeType *VrmlNodeMetadataString::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MetadataString", creator);
    }

    VrmlNodeMetadata::defineType(t); // Parent class

    t->addExposedField("value", VrmlField::MFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeMetadataString::nodeType() const { return defineType(0); }

VrmlNodeMetadataString::VrmlNodeMetadataString(VrmlScene *scene)
    : VrmlNodeMetadata(scene)
{
}

VrmlNodeMetadataString::~VrmlNodeMetadataString()
{
}

VrmlNode *VrmlNodeMetadataString::cloneMe() const
{
    return new VrmlNodeMetadataString(*this);
}

std::ostream &VrmlNodeMetadataString::printFields(std::ostream &os, int indent)
{
    if (!d_value.get())
        PRINT_FIELD(value);

    VrmlNodeMetadata::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadataString::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    if
        TRY_FIELD(value, MFString)
    else
        VrmlNodeMetadata::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadataString::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "value") == 0)
        return &d_value;
    return VrmlNodeMetadata::getField(fieldName);
}

VrmlNodeMetadataString *VrmlNodeMetadataString::toMetadataString() const
{
    return (VrmlNodeMetadataString *)this;
}
