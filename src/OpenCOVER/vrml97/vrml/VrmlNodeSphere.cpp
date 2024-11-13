/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSphere.cpp

#include "VrmlNodeSphere.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeSphere::initFields(VrmlNodeSphere *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); 
    initFieldsHelper(node, t, field("radius", node->d_radius));
}

const char *VrmlNodeSphere::name() { return "Sphere"; }

VrmlNodeSphere::VrmlNodeSphere(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
    , d_radius(1.0)
{
}

Viewer::Object VrmlNodeSphere::insertGeometry(Viewer *viewer)
{
    return viewer->insertSphere(d_radius.get());
}
