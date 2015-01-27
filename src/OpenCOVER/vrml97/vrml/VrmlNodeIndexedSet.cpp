/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIndexedSet.cpp

#include "VrmlNodeIndexedSet.h"

#include "VrmlNodeType.h"

using namespace vrml;

// Define the built in VrmlNodeType:: "Indexed*Set" fields

VrmlNodeType *VrmlNodeIndexedSet::defineType(VrmlNodeType *t)
{
    VrmlNodeColoredSet::defineType(t); // Parent class

    t->addEventIn("set_colorIndex", VrmlField::MFINT32);
    t->addEventIn("set_coordIndex", VrmlField::MFINT32);
    t->addField("colorIndex", VrmlField::MFINT32);
    t->addField("coordIndex", VrmlField::MFINT32);

    return t;
}

VrmlNodeIndexedSet::VrmlNodeIndexedSet(VrmlScene *scene)
    : VrmlNodeColoredSet(scene)
{
}

VrmlNodeIndexedSet::~VrmlNodeIndexedSet()
{
}

bool VrmlNodeIndexedSet::isModified() const
{
    return (d_modified);
}

void VrmlNodeIndexedSet::clearFlags()
{
    VrmlNode::clearFlags();
}

void VrmlNodeIndexedSet::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    nodeStack.pop_front();
}

void VrmlNodeIndexedSet::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    nodeStack.pop_front();
}

std::ostream &VrmlNodeIndexedSet::printFields(std::ostream &os, int indent)
{
    if (d_colorIndex.size() > 0)
        PRINT_FIELD(colorIndex);
    if (d_coordIndex.size() > 0)
        PRINT_FIELD(coordIndex);
    return os;
}

const VrmlMFInt &VrmlNodeIndexedSet::getCoordIndex() const
{
    return d_coordIndex;
}

// LarryD Feb 18/99
const VrmlMFInt &VrmlNodeIndexedSet::getColorIndex() const
{
    return d_colorIndex;
}

// Set the value of one of the node fields.

void VrmlNodeIndexedSet::setField(const char *fieldName,
                                  const VrmlField &fieldValue)
{
    if
        TRY_FIELD(colorIndex, MFInt)
    else if
        TRY_FIELD(coordIndex, MFInt)
    else
        VrmlNodeColoredSet::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeIndexedSet::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "colorIndex") == 0)
        return &d_colorIndex;
    else if (strcmp(fieldName, "coordIndex") == 0)
        return &d_coordIndex;

    return VrmlNodeColoredSet::getField(fieldName);
}
