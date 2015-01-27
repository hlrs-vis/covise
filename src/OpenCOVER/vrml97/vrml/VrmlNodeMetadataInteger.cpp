/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataInteger.cpp

#include "VrmlNodeMetadataInteger.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeMetadataInteger(s); }

// Define the built in VrmlNodeType:: "MetadataInteger" fields

VrmlNodeType *VrmlNodeMetadataInteger::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MetadataInteger", creator);
    }

    VrmlNodeMetadata::defineType(t); // Parent class

    t->addExposedField("value", VrmlField::MFINT32);

    return t;
}

VrmlNodeType *VrmlNodeMetadataInteger::nodeType() const { return defineType(0); }

VrmlNodeMetadataInteger::VrmlNodeMetadataInteger(VrmlScene *scene)
    : VrmlNodeMetadata(scene)
{
}

VrmlNodeMetadataInteger::~VrmlNodeMetadataInteger()
{
}

VrmlNode *VrmlNodeMetadataInteger::cloneMe() const
{
    return new VrmlNodeMetadataInteger(*this);
}

std::ostream &VrmlNodeMetadataInteger::printFields(std::ostream &os, int indent)
{
    if (!d_value.get())
        PRINT_FIELD(value);

    VrmlNodeMetadata::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadataInteger::setField(const char *fieldName,
                                       const VrmlField &fieldValue)
{
    if
        TRY_FIELD(value, MFInt)
    else
        VrmlNodeMetadata::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadataInteger::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "value") == 0)
        return &d_value;
    return VrmlNodeMetadata::getField(fieldName);
}

VrmlNodeMetadataInteger *VrmlNodeMetadataInteger::toMetadataInteger() const
{
    return (VrmlNodeMetadataInteger *)this;
}
