/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCylinder.cpp

#include "VrmlNodeCylinder.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeCylinder::initFields(VrmlNodeCylinder *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     field("bottom", node->d_bottom),
                     field("height", node->d_height),
                     field("radius", node->d_radius),
                     field("side", node->d_side),
                     field("top", node->d_top));
}

const char *VrmlNodeCylinder::name() { return "Cylinder"; }


VrmlNodeCylinder::VrmlNodeCylinder(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
    , d_bottom(true)
    , d_height(2.0)
    , d_radius(1.0)
    , d_side(true)
    , d_top(true)
{
}

Viewer::Object VrmlNodeCylinder::insertGeometry(Viewer *viewer)
{
    return viewer->insertCylinder(d_height.get(),
                                  d_radius.get(),
                                  d_bottom.get(),
                                  d_side.get(),
                                  d_top.get());
}
