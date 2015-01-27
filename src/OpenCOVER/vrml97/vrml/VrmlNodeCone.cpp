/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCone.cpp

#include "VrmlNodeCone.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

//  Return a new VrmlNodeCone
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeCone(s); }

// Define the built in VrmlNodeType:: "Cone" fields

VrmlNodeType *VrmlNodeCone::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define type once.
        t = st = new VrmlNodeType("Cone", creator);
    }

    VrmlNodeGeometry::defineType(t); // Parent class
    t->addField("bottom", VrmlField::SFBOOL);
    t->addField("bottomRadius", VrmlField::SFFLOAT);
    t->addField("height", VrmlField::SFFLOAT);
    t->addField("side", VrmlField::SFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeCone::nodeType() const { return defineType(0); }

VrmlNodeCone::VrmlNodeCone(VrmlScene *scene)
    : VrmlNodeGeometry(scene)
    , d_bottom(true)
    , d_bottomRadius(1.0)
    , d_height(2.0)
    , d_side(true)
{
}

VrmlNodeCone::~VrmlNodeCone()
{
}

VrmlNode *VrmlNodeCone::cloneMe() const
{
    return new VrmlNodeCone(*this);
}

std::ostream &VrmlNodeCone::printFields(std::ostream &os, int indent)
{
    if (!d_bottom.get())
        PRINT_FIELD(bottom);
    if (d_bottomRadius.get() != 1.0)
        PRINT_FIELD(bottomRadius);
    if (d_height.get() != 2.0)
        PRINT_FIELD(height);
    if (!d_side.get())
        PRINT_FIELD(side);

    return os;
}

Viewer::Object VrmlNodeCone::insertGeometry(Viewer *viewer)
{
    return viewer->insertCone(d_height.get(),
                              d_bottomRadius.get(),
                              d_bottom.get(),
                              d_side.get());
}

// Set the value of one of the node fields.

void VrmlNodeCone::setField(const char *fieldName,
                            const VrmlField &fieldValue)
{
    if
        TRY_FIELD(bottom, SFBool)
    else if
        TRY_FIELD(bottomRadius, SFFloat)
    else if
        TRY_FIELD(height, SFFloat)
    else if
        TRY_FIELD(side, SFBool)
    else
        VrmlNodeGeometry::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeCone::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "bottom") == 0)
        return &d_bottom;
    else if (strcmp(fieldName, "bottomRadius") == 0)
        return &d_bottomRadius;
    else if (strcmp(fieldName, "height") == 0)
        return &d_height;
    else if (strcmp(fieldName, "side") == 0)
        return &d_side;

    return VrmlNodeGeometry::getField(fieldName);
}

VrmlNodeCone *VrmlNodeCone::toCone() const //LarryD Mar 08/99
{
    return (VrmlNodeCone *)this;
}
