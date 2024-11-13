/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBox.cpp

#include "VrmlNodeBox.h"

#include "VrmlNodeType.h"
#include "MathUtils.h"
#include "Viewer.h"
#include <math.h>

using namespace vrml;

void VrmlNodeBox::initFields(VrmlNodeBox *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     field("size", node->d_size));
}

const char *VrmlNodeBox::name()
{
    return "Box";
}

VrmlNodeBox::VrmlNodeBox(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
    , d_size(2.0, 2.0, 2.0)
{
}

Viewer::Object VrmlNodeBox::insertGeometry(Viewer *viewer)
{
    return viewer->insertBox(d_size.x(), d_size.y(), d_size.z());
}

const VrmlSFVec3f &VrmlNodeBox::getSize() const // LarryD Mar 08/99
{
    return d_size;
}
