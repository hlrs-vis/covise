/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCone.cpp

#include "VrmlNodeCone.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeCone::initFields(VrmlNodeCone *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                        field("bottom", node->d_bottom),
                        field("bottomRadius", node->d_bottomRadius),
                        field("height", node->d_height),
                        field("side", node->d_side));
}

const char *VrmlNodeCone::name() { return "Cone"; }

VrmlNodeCone::VrmlNodeCone(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
    , d_bottom(true)
    , d_bottomRadius(1.0)
    , d_height(2.0)
    , d_side(true)
{
}

Viewer::Object VrmlNodeCone::insertGeometry(Viewer *viewer)
{
    return viewer->insertCone(d_height.get(),
                              d_bottomRadius.get(),
                              d_bottom.get(),
                              d_side.get());
}

VrmlNodeCone *VrmlNodeCone::toCone() const //LarryD Mar 08/99
{
    return (VrmlNodeCone *)this;
}
