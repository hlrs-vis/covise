/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureCoordinateGenerator.cpp

#include "VrmlNodeTextureCoordinateGenerator.h"
#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeTextureCoordinateGenerator::initFields(VrmlNodeTextureCoordinateGenerator *node, VrmlNodeType *t)
{
    VrmlNode::initFieldsHelper(node, t, 
        exposedField("mode", node->d_mode), 
        exposedField("parameter", node->d_parameter));
}

const char *VrmlNodeTextureCoordinateGenerator::name()
{
    return "TextureCoordinateGenerator";
}

VrmlNodeTextureCoordinateGenerator::VrmlNodeTextureCoordinateGenerator(VrmlScene *scene)
    : VrmlNode(scene, name())
    , d_mode("SPHERE")
{
}
