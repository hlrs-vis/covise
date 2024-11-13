/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColor.cpp

#include "VrmlNodeColor.h"

#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeColor::initFields(VrmlNodeColor *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("color", node->d_color));
}

const char *VrmlNodeColor::name() { return "Color"; }

VrmlNodeColor::VrmlNodeColor(VrmlScene *scene)
    : VrmlNode(scene, name())
{
}
