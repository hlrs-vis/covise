/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCoordinate.cpp

#include "VrmlNodeCoordinate.h"
#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeCoordinate::initFields(VrmlNodeCoordinate *node, VrmlNodeType *t)
{
    VrmlNode::initFieldsHelper(node, t,
                                       exposedField("point", node->d_point));
}

const char *VrmlNodeCoordinate::name() { return "Coordinate"; }

VrmlNodeCoordinate::VrmlNodeCoordinate(VrmlScene *scene)
    : VrmlNode(scene, name())
{
}

int VrmlNodeCoordinate::getNumberCoordinates()
{
    if (d_point.get())
        return d_point.size() / 3;
    return 0;
}
