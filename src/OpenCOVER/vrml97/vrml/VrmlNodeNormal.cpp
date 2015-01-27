/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNormal.cpp

#include "VrmlNodeNormal.h"
#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeNormal(s); }

// Define the built in VrmlNodeType:: "Normal" fields

VrmlNodeType *VrmlNodeNormal::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Normal", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("vector", VrmlField::MFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeNormal::nodeType() const { return defineType(0); }

VrmlNodeNormal::VrmlNodeNormal(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeNormal::~VrmlNodeNormal()
{
}

VrmlNode *VrmlNodeNormal::cloneMe() const
{
    return new VrmlNodeNormal(*this);
}

VrmlNodeNormal *VrmlNodeNormal::toNormal() const
{
    return (VrmlNodeNormal *)this;
}

std::ostream &VrmlNodeNormal::printFields(std::ostream &os, int indent)
{
    if (d_vector.size() > 0)
        PRINT_FIELD(vector);
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeNormal::setField(const char *fieldName,
                              const VrmlField &fieldValue)
{
    if
        TRY_FIELD(vector, MFVec3f)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeNormal::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "vector") == 0)
        return &d_vector;

    return VrmlNode::getField(fieldName);
}
