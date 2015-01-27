/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadata.cpp

#include "VrmlNodeMetadata.h"

#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "System.h"

using namespace vrml;

//  Metadata factory.

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeMetadata(s);
}

// Define the built in VrmlNodeType:: "Metadata" fields

VrmlNodeType *VrmlNodeMetadata::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Metadata", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addField("name", VrmlField::SFSTRING);
    t->addField("reference", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeMetadata::nodeType() const { return defineType(0); }

VrmlNodeMetadata::VrmlNodeMetadata(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeMetadata::~VrmlNodeMetadata()
{
}

VrmlNode *VrmlNodeMetadata::cloneMe() const
{
    return new VrmlNodeMetadata(*this);
}

std::ostream &VrmlNodeMetadata::printFields(std::ostream &os, int indent)
{
    if (d_name.get())
        PRINT_FIELD(name);
    if (d_reference.get())
        PRINT_FIELD(reference);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadata::setField(const char *fieldName,
                                const VrmlField &fieldValue)
{
    if
        TRY_FIELD(name, SFString)
    else if
        TRY_FIELD(reference, SFString)
    else
        VrmlNode::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadata::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "name") == 0)
        return &d_name;
    else if (strcmp(fieldName, "reference") == 0)
        return &d_reference;

    return VrmlNode::getField(fieldName);
}
