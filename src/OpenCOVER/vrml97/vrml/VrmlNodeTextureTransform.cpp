/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureTransform.cpp

#include "VrmlNodeTextureTransform.h"
#include "VrmlNodeType.h"
#include "MathUtils.h"
#include "Viewer.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeTextureTransform(s);
}

// Define the built in VrmlNodeType:: "TextureTransform" fields

VrmlNodeType *VrmlNodeTextureTransform::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("TextureTransform", creator);
    }

    VrmlNode::defineType(t); // Parent class
    t->addExposedField("center", VrmlField::SFVEC2F);
    t->addExposedField("rotation", VrmlField::SFFLOAT);
    t->addExposedField("scale", VrmlField::SFVEC2F);
    t->addExposedField("translation", VrmlField::SFVEC2F);

    return t;
}

VrmlNodeType *VrmlNodeTextureTransform::nodeType() const
{
    return defineType(0);
}

VrmlNodeTextureTransform::VrmlNodeTextureTransform(VrmlScene *scene)
    : VrmlNode(scene)
    , d_center(0.0, 0.0)
    , d_rotation(0.0)
    , d_scale(1.0, 1.0)
    , d_translation(0.0, 0.0)
{
}

VrmlNodeTextureTransform::~VrmlNodeTextureTransform()
{
}

VrmlNode *VrmlNodeTextureTransform::cloneMe() const
{
    return new VrmlNodeTextureTransform(*this);
}

VrmlNodeTextureTransform *VrmlNodeTextureTransform::toTextureTransform() const
{
    return (VrmlNodeTextureTransform *)this;
}

std::ostream &VrmlNodeTextureTransform::printFields(std::ostream &os, int indent)
{
    if (!FPZERO(d_center.x()) || !FPZERO(d_center.y()))
        PRINT_FIELD(center);

    if (!FPZERO(d_rotation.get()))
        PRINT_FIELD(rotation);

    if (!FPEQUAL(d_scale.x(), 1.0) || !FPEQUAL(d_scale.y(), 1.0))
        PRINT_FIELD(scale);
    if (!FPZERO(d_translation.x()) || !FPZERO(d_translation.y()))
        PRINT_FIELD(translation);

    return os;
}

void VrmlNodeTextureTransform::render(Viewer *viewer)
{
    viewer->setTextureTransform(d_center.get(),
                                d_rotation.get(),
                                d_scale.get(),
                                d_translation.get());
    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeTextureTransform::setField(const char *fieldName,
                                        const VrmlField &fieldValue)
{
    if
        TRY_FIELD(center, SFVec2f)
    else if
        TRY_FIELD(rotation, SFFloat)
    else if
        TRY_FIELD(scale, SFVec2f)
    else if
        TRY_FIELD(translation, SFVec2f)
    else
        VrmlNode::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTextureTransform::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "center") == 0)
        return &d_center;
    else if (strcmp(fieldName, "rotation") == 0)
        return &d_rotation;
    else if (strcmp(fieldName, "scale") == 0)
        return &d_scale;
    else if (strcmp(fieldName, "translation") == 0)
        return &d_translation;

    return VrmlNode::getField(fieldName);
}
