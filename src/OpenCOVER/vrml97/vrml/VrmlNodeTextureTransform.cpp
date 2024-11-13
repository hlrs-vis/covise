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

void VrmlNodeTextureTransform::initFields(VrmlNodeTextureTransform *node, VrmlNodeType *t)
{
    VrmlNode::initFieldsHelper(node, t,
                                       exposedField("center", node->d_center),
                                       exposedField("rotation", node->d_rotation),
                                       exposedField("scale", node->d_scale),
                                       exposedField("translation", node->d_translation));
}

const char *VrmlNodeTextureTransform::name()
{
    return "TextureTransform";
}

VrmlNodeTextureTransform::VrmlNodeTextureTransform(VrmlScene *scene)
    : VrmlNode(scene, name())
    , d_center(0.0, 0.0)
    , d_rotation(0.0)
    , d_scale(1.0, 1.0)
    , d_translation(0.0, 0.0)
{
}

void VrmlNodeTextureTransform::render(Viewer *viewer)
{
    viewer->setTextureTransform(d_center.get(),
                                d_rotation.get(),
                                d_scale.get(),
                                d_translation.get());
    clearModified();
}
