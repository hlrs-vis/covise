/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTransform.cpp

#include "VrmlNodeTransform.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeTransform(s); }

// Define the built in VrmlNodeType:: "Transform" fields

void VrmlNodeTransform::initFields(VrmlNodeTransform *node, VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
        exposedField("center", node->d_center),
        exposedField("rotation", node->d_rotation),
        exposedField("scale", node->d_scale),
        exposedField("scaleOrientation", node->d_scaleOrientation),
        exposedField("translation", node->d_translation)
    );
}

const char *VrmlNodeTransform::name() { return "Transform"; }

VrmlNodeTransform::VrmlNodeTransform(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
    , d_center(0.0, 0.0, 0.0)
    , d_rotation(0.0, 0.0, 1.0, 0.0)
    , d_scale(1.0, 1.0, 1.0)
    , d_scaleOrientation(0.0, 0.0, 1.0, 0.0)
    , d_translation(0.0, 0.0, 0.0)
    , d_xformObject(0)
{
    d_modified = true;
}

//LarryD Feb24/99
VrmlNodeTransform *VrmlNodeTransform::toTransform() const
{
    return (VrmlNodeTransform *)this;
}

// LarryD Feb 18/99
const VrmlSFVec3f &VrmlNodeTransform::getCenter() const
{
    return d_center;
}

//LarryD Feb 24/99
const VrmlSFRotation &VrmlNodeTransform::getRotation() const
{
    return d_rotation;
}

//LarryD Feb 24/99
const VrmlSFVec3f &VrmlNodeTransform::getScale() const
{
    return d_scale;
}

//LarryD Feb 24/99
const VrmlSFRotation &VrmlNodeTransform::getScaleOrientation() const
{
    return d_scaleOrientation;
}

//LarryD Feb 24/99
const VrmlSFVec3f &VrmlNodeTransform::getTranslation() const
{
    return d_translation;
}

void VrmlNodeTransform::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_xformObject && isModified())
    {
        viewer->removeObject(d_xformObject);
        d_xformObject = 0;
    }
    checkAndRemoveNodes(viewer);
    if (d_xformObject)
    {
        viewer->insertReference(d_xformObject);
    }
    else if (d_children.size() > 0)
    {
        d_xformObject = viewer->beginObject(name(), 0, this);
            // Apply transforms
            viewer->setTransform(d_center.get(),
                d_rotation.get(),
                d_scale.get(),
                d_scaleOrientation.get(),
                d_translation.get(), d_modified);

        // Render children
        VrmlNodeGroup::render(viewer);

            // Reverse transforms (for immediate mode/no matrix stack renderer)
            viewer->unsetTransform(d_center.get(),
                d_rotation.get(),
                d_scale.get(),
                d_scaleOrientation.get(),
                d_translation.get());
        viewer->endObject();
    }

    clearModified();
}

// Cache a pointer to (one of the) parent transforms for proper
// rendering of bindables.

void VrmlNodeTransform::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;

    int i, n = d_children.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_children[i];
        if (kid)
            kid->accumulateTransform(this);
    }
}

void VrmlNodeTransform::inverseTransform(Viewer *viewer)
{
    VrmlNode *parentTransform = getParentTransform();
    if (parentTransform)
        parentTransform->inverseTransform(viewer);

    // Build the inverse
    float trans[3] = {
        -d_translation.x(),
        -d_translation.y(),
        -d_translation.z()
    };
    float rot[4] = {
        d_rotation.x(),
        d_rotation.y(),
        d_rotation.z(),
        -d_rotation.r()
    };

    // Invert scale (1/x)
    float scale[3] = { d_scale.x(), d_scale.y(), d_scale.z() };
    if (!FPZERO(scale[0]))
        scale[0] = 1.0f / scale[0];
    if (!FPZERO(scale[1]))
        scale[1] = 1.0f / scale[1];
    if (!FPZERO(scale[2]))
        scale[2] = 1.0f / scale[2];

    // Apply transforms (this may need to be broken into separate
    // calls to perform the ops in reverse order...)
    viewer->setTransform(d_center.get(),
                         rot,
                         scale,
                         d_scaleOrientation.get(),
                         trans, true);
}

void VrmlNodeTransform::inverseTransform(double *m)
{
    VrmlNode *parentTransform = getParentTransform();
    if (strncmp(name(), "StaticCave", 10) == 0)
    {
        Midentity(m);
    }
    else
    {
        if (parentTransform)
        {
            parentTransform->inverseTransform(m);
        }
        else
            Midentity(m);
    }

    // Invert this transform
    float rot[4] = {
        d_rotation.x(),
        d_rotation.y(),
        d_rotation.z(),
        -d_rotation.r()
    };
    float nsrot[4] = {
        d_scaleOrientation.x(),
        d_scaleOrientation.y(),
        d_scaleOrientation.z(),
        -d_scaleOrientation.r()
    };
    float srot[4] = {
        d_scaleOrientation.x(),
        d_scaleOrientation.y(),
        d_scaleOrientation.z(),
        d_scaleOrientation.r()
    };
    float trans[3] = { (-d_translation.x()), (-d_translation.y()), (-d_translation.z()) };
    float center[3] = { (d_center.x()), (d_center.y()), (d_center.z()) };
    float ncenter[3] = { (-d_center.x()), (-d_center.y()), (-d_center.z()) };
    double M[16];

    Mtrans(M, trans);
    MM(m, M);
    Mtrans(M, ncenter);
    MM(m, M);

    // Invert scale (1/x)
    float scale[3] = { d_scale.x(), d_scale.y(), d_scale.z() };
    if (!FPZERO(scale[0]))
        scale[0] = 1.0f / scale[0];
    if (!FPZERO(scale[1]))
        scale[1] = 1.0f / scale[1];
    if (!FPZERO(scale[2]))
        scale[2] = 1.0f / scale[2];

    Mrotation(M, rot);
    MM(m, M);

    if (!FPEQUAL(scale[0], 1.0) || !FPEQUAL(scale[1], 1.0) || !FPEQUAL(scale[2], 1.0))
    {
        //fprintf(stderr, "invTr: scale\n");
        if (!FPZERO(srot[3]))
        {
            //fprintf(stderr, "invTr: rot\n");
            Mrotation(M, nsrot);
            MM(m, M);
            Mscale(M, scale);
            MM(m, M);
            Mrotation(M, srot);
            MM(m, M);
        }
        else
        {
            Mscale(M, scale);
            MM(m, M);
        }
    }
    Mtrans(M, center);
    MM(m, M);
}
