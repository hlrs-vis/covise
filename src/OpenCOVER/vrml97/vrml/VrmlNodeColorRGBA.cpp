/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColorRGBA.cpp

#include "VrmlNodeColorRGBA.h"

#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeColorRGBA::initFields(VrmlNodeColorRGBA *node, VrmlNodeType *t)
{
    if (!t)
        t = node->nodeType();

    VrmlNodeTemplate::initFieldsHelper(node, t,
                                       exposedField("color", node->d_color));
}

const char *VrmlNodeColorRGBA::name() { return "ColorRGBA"; }

VrmlNodeColorRGBA::VrmlNodeColorRGBA(VrmlScene *scene)
    : VrmlNodeTemplate(scene, name())
{
}

VrmlNodeColorRGBA *VrmlNodeColorRGBA::toColorRGBA() const
{
    return (VrmlNodeColorRGBA *)this;
}
