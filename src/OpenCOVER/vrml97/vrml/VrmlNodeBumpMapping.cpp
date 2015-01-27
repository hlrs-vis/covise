/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBumpMapping.cpp
//

#include "VrmlNodeBumpMapping.h"
#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeBumpMapping(scene);
}

// Define the built in VrmlNodeType:: "BumpMapping" fields

VrmlNodeType *VrmlNodeBumpMapping::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("BumpMapping", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeBumpMapping::nodeType() const { return defineType(0); }

VrmlNodeBumpMapping::VrmlNodeBumpMapping(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
}

VrmlNodeBumpMapping::~VrmlNodeBumpMapping()
{
}

VrmlNode *VrmlNodeBumpMapping::cloneMe() const
{
    return new VrmlNodeBumpMapping(*this);
}

VrmlNodeBumpMapping *VrmlNodeBumpMapping::toBumpMapping() const
{
    return (VrmlNodeBumpMapping *)this;
}

void VrmlNodeBumpMapping::addToScene(VrmlScene *, const char * /*rel*/)
{
}

std::ostream &VrmlNodeBumpMapping::printFields(std::ostream &os, int)
{
    return os;
}

void VrmlNodeBumpMapping::render(Viewer *viewer)
{
    viewer->insertBumpMapping();

    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeBumpMapping::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeBumpMapping::getField(const char *fieldName) const
{
    return VrmlNodeChild::getField(fieldName);
}
