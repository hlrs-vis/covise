/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataBoolean.cpp

#include "VrmlNodeMetadataBoolean.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeMetadataBoolean(s); }

// Define the built in VrmlNodeType:: "MetadataBoolean" fields

VrmlNodeType *VrmlNodeMetadataBoolean::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MetadataBoolean", creator);
    }

    VrmlNodeMetadata::defineType(t); // Parent class

    t->addExposedField("value", VrmlField::MFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeMetadataBoolean::nodeType() const { return defineType(0); }

VrmlNodeMetadataBoolean::VrmlNodeMetadataBoolean(VrmlScene *scene)
    : VrmlNodeMetadata(scene)
{
}

VrmlNodeMetadataBoolean::~VrmlNodeMetadataBoolean()
{
}

VrmlNode *VrmlNodeMetadataBoolean::cloneMe() const
{
    return new VrmlNodeMetadataBoolean(*this);
}

std::ostream &VrmlNodeMetadataBoolean::printFields(std::ostream &os, int indent)
{
    if (!d_value.get())
        PRINT_FIELD(value);

    VrmlNodeMetadata::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadataBoolean::setField(const char *fieldName,
                                       const VrmlField &fieldValue)
{
    if
        TRY_FIELD(value, MFBool)
    else
        VrmlNodeMetadata::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadataBoolean::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "value") == 0)
        return &d_value;
    return VrmlNodeMetadata::getField(fieldName);
}

VrmlNodeMetadataBoolean *VrmlNodeMetadataBoolean::toMetadataBoolean() const
{
    return (VrmlNodeMetadataBoolean *)this;
}
