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
                     field("bottom", &VrmlNodeCylinder::d_bottom),
                     field("height", &VrmlNodeCylinder::d_height),
                     field("radius", &VrmlNodeCylinder::d_radius),
                     field("side", &VrmlNodeCylinder::d_side),
                     field("top", &VrmlNodeCylinder::d_top));
}

const char *VrmlNodeCylinder::typeName() { return "Cylinder"; }


VrmlNodeCylinder::VrmlNodeCylinder(VrmlScene *scene)
    : VrmlNodeGeometry(scene, typeName())
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
