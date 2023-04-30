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

VrmlNodeType *VrmlNodeTransform::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Transform", creator);
    }

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("center", VrmlField::SFVEC3F);
    t->addExposedField("rotation", VrmlField::SFROTATION);
    t->addExposedField("scale", VrmlField::SFVEC3F);
    t->addExposedField("scaleOrientation", VrmlField::SFROTATION);
    t->addExposedField("translation", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeTransform::nodeType() const { return defineType(0); }

VrmlNodeTransform::VrmlNodeTransform(VrmlScene *scene)
    : VrmlNodeGroup(scene)
    , d_center(0.0, 0.0, 0.0)
    , d_rotation(0.0, 0.0, 1.0, 0.0)
    , d_scale(1.0, 1.0, 1.0)
    , d_scaleOrientation(0.0, 0.0, 1.0, 0.0)
    , d_translation(0.0, 0.0, 0.0)
    , d_xformObject(0)
{
    d_modified = true;
}

VrmlNodeTransform::~VrmlNodeTransform()
{
    // delete d_xformObject...
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

VrmlNode *VrmlNodeTransform::cloneMe() const
{
    return new VrmlNodeTransform(*this);
}

std::ostream &VrmlNodeTransform::printFields(std::ostream &os, int indent)
{
    if (!FPZERO(d_center.x()) || !FPZERO(d_center.y()) || !FPZERO(d_center.z()))
        PRINT_FIELD(center);
    if (!FPZERO(d_rotation.x()) || !FPZERO(d_rotation.y()) || !FPEQUAL(d_rotation.z(), 1.0) || !FPZERO(d_rotation.r()))
        PRINT_FIELD(rotation);
    if (!FPEQUAL(d_scale.x(), 1.0) || !FPEQUAL(d_scale.y(), 1.0) || !FPEQUAL(d_scale.z(), 1.0))
        PRINT_FIELD(scale);
    if (!FPZERO(d_scaleOrientation.x()) || !FPZERO(d_scaleOrientation.y()) || !FPEQUAL(d_scaleOrientation.z(), 1.0) || !FPZERO(d_scaleOrientation.r()))
        PRINT_FIELD(scaleOrientation);
    if (!FPZERO(d_translation.x()) || !FPZERO(d_translation.y()) || !FPZERO(d_translation.z()))
        PRINT_FIELD(translation);

    VrmlNodeGroup::printFields(os, indent);
    return os;
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

// Set the value of one of the node fields.

void VrmlNodeTransform::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(center, SFVec3f)
    else if
        TRY_FIELD(rotation, SFRotation)
    else if
        TRY_FIELD(scale, SFVec3f)
    else if
        TRY_FIELD(scaleOrientation, SFRotation)
    else if
        TRY_FIELD(translation, SFVec3f)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeTransform::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "center") == 0)
        return &d_center;
    else if (strcmp(fieldName, "rotation") == 0)
        return &d_rotation;
    else if (strcmp(fieldName, "scale") == 0)
        return &d_scale;
    else if (strcmp(fieldName, "scaleOrientation") == 0)
        return &d_scaleOrientation;
    else if (strcmp(fieldName, "translation") == 0)
        return &d_translation;

    return VrmlNodeGroup::getField(fieldName);
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
