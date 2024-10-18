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
void VrmlNodeBillboard::initFields(VrmlNodeBillboard *node, vrml::VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("axisOfRotation", node->d_axisOfRotation));
    VrmlNodeGroup::initFields(node, t);
}

const char *VrmlNodeBillboard::name() { return "Billboard"; }

VrmlNodeBillboard::VrmlNodeBillboard(VrmlScene *scene, const std::string &name)
    : VrmlNodeGroup(scene, name == ""? this->name() : name)
    , d_axisOfRotation(0.0, 1.0, 0.0)
    , d_xformObject(0)
{
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
