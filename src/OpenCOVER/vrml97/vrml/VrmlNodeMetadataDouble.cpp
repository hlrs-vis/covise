/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataDouble.cpp

#include "VrmlNodeMetadataDouble.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeMetadataDouble(s); }

// Define the built in VrmlNodeType:: "MetadataDouble" fields

VrmlNodeType *VrmlNodeMetadataDouble::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MetadataDouble", creator);
    }

    VrmlNodeMetadata::defineType(t); // Parent class

    t->addExposedField("value", VrmlField::MFDOUBLE);

    return t;
}

VrmlNodeType *VrmlNodeMetadataDouble::nodeType() const { return defineType(0); }

VrmlNodeMetadataDouble::VrmlNodeMetadataDouble(VrmlScene *scene)
    : VrmlNodeMetadata(scene)
{
}

VrmlNodeMetadataDouble::~VrmlNodeMetadataDouble()
{
}

VrmlNode *VrmlNodeMetadataDouble::cloneMe() const
{
    return new VrmlNodeMetadataDouble(*this);
}

std::ostream &VrmlNodeMetadataDouble::printFields(std::ostream &os, int indent)
{
    if (!d_value.get())
        PRINT_FIELD(value);

    VrmlNodeMetadata::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadataDouble::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    if
        TRY_FIELD(value, MFDouble)
    else
        VrmlNodeMetadata::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadataDouble::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "value") == 0)
        return &d_value;
    return VrmlNodeMetadata::getField(fieldName);
}

VrmlNodeMetadataDouble *VrmlNodeMetadataDouble::toMetadataDouble() const
{
    return (VrmlNodeMetadataDouble *)this;
}
