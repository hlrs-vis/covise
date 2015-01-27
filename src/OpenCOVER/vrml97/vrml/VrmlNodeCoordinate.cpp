/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCoordinate.cpp

#include "VrmlNodeCoordinate.h"
#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeCoordinate(s); }

// Define the built in VrmlNodeType:: "Coordinate" fields

VrmlNodeType *VrmlNodeCoordinate::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Coordinate", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("point", VrmlField::MFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeCoordinate::nodeType() const { return defineType(0); }

VrmlNodeCoordinate::VrmlNodeCoordinate(VrmlScene *scene)
    : VrmlNode(scene)
{
}

VrmlNodeCoordinate::~VrmlNodeCoordinate()
{
}

VrmlNode *VrmlNodeCoordinate::cloneMe() const
{
    return new VrmlNodeCoordinate(*this);
}

VrmlNodeCoordinate *VrmlNodeCoordinate::toCoordinate() const
{
    return (VrmlNodeCoordinate *)this;
}

std::ostream &VrmlNodeCoordinate::printFields(std::ostream &os, int indent)
{
    if (d_point.size() > 0)
        PRINT_FIELD(point);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCoordinate::setField(const char *fieldName,
                                  const VrmlField &fieldValue)
{
    if
        TRY_FIELD(point, MFVec3f)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeCoordinate::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "point") == 0)
        return &d_point;

    return VrmlNode::getField(fieldName);
}

int VrmlNodeCoordinate::getNumberCoordinates()
{
    if (d_point.get())
        return d_point.size() / 3;
    return 0;
}
