/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBillboard.cpp

#include "VrmlNodeBillboard.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"
#include <math.h>

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeBillboard(s); }

// Define the built in VrmlNodeType:: "Billboard" fields

VrmlNodeType *VrmlNodeBillboard::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Billboard", creator);
    }

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("axisOfRotation", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeBillboard::nodeType() const { return defineType(0); }

VrmlNodeBillboard::VrmlNodeBillboard(VrmlScene *scene)
    : VrmlNodeGroup(scene)
    , d_axisOfRotation(0.0, 1.0, 0.0)
    , d_xformObject(0)
{
}

VrmlNodeBillboard::~VrmlNodeBillboard()
{
    // delete d_xformObject...
}

VrmlNode *VrmlNodeBillboard::cloneMe() const
{
    return new VrmlNodeBillboard(*this);
}

std::ostream &VrmlNodeBillboard::printFields(std::ostream &os, int indent)
{
    if (!FPZERO(d_axisOfRotation.x()) || !FPZERO(d_axisOfRotation.y()) || !FPZERO(d_axisOfRotation.z()))
        PRINT_FIELD(axisOfRotation);

    VrmlNodeGroup::printFields(os, indent);
    return os;
}

void VrmlNodeBillboard::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_xformObject && isModified())
    {
        viewer->removeObject(d_xformObject);
        d_xformObject = 0;
    }

    if (d_xformObject)
        viewer->insertReference(d_xformObject);

    else if (d_children.size() > 0)
    {
        d_xformObject = viewer->beginObject(name(), 0, this);

        viewer->setBillboardTransform(d_axisOfRotation.get());

        // Render children
        VrmlNodeGroup::render(viewer);

        viewer->unsetBillboardTransform(d_axisOfRotation.get());

        viewer->endObject();
    }

    clearModified();
}

// Cache a pointer to (one of the) parent transforms for proper
// rendering of bindables.

void VrmlNodeBillboard::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;

    int i, n = d_children.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_children[i];
        kid->accumulateTransform(this);
    }
}

VrmlNode *VrmlNodeBillboard::getParentTransform() { return d_parentTransform; }

void VrmlNodeBillboard::inverseTransform(Viewer *viewer)
{
    VrmlNode *parentTransform = getParentTransform();
    if (parentTransform)
        parentTransform->inverseTransform(viewer);

    // Apply inverted bb transforms...
    //viewer->setBillboardTransform( d_axisOfRotation.get() );
}

void VrmlNodeBillboard::inverseTransform(double *m)
{
    VrmlNode *parentTransform = getParentTransform();
    if (parentTransform)
        parentTransform->inverseTransform(m);
    else
        Midentity(m);

    // Invert bb transform...
    // ...
}

// Set the value of one of the node fields.

void VrmlNodeBillboard::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(axisOfRotation, SFVec3f)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeBillboard::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "axisOfRotation") == 0)
        return &d_axisOfRotation;

    return VrmlNodeGroup::getField(fieldName);
}
