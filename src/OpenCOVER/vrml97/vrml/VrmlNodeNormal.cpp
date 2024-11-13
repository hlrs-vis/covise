/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNormal.cpp

#include "VrmlNodeNormal.h"
#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeNormal::initFields(VrmlNodeNormal *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t, exposedField("vector", node->d_vector));
}

const char *VrmlNodeNormal::name() { return "Normal"; }

VrmlNodeNormal::VrmlNodeNormal(VrmlScene *scene)
    : VrmlNode(scene, name())
{
}
