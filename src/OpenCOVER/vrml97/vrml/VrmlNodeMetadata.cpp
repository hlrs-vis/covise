/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadata.cpp

#include "VrmlNodeMetadata.h"

#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "System.h"

using namespace vrml;

void VrmlNodeMetadata::initFields(VrmlNodeMetadata *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t, 
                        field("name", node->d_name),
                        field("reference", node->d_reference));
}

const char *VrmlNodeMetadata::name() { return "Metadata"; }

VrmlNodeMetadata::VrmlNodeMetadata(VrmlScene *scene, const std::string &name)
    : VrmlNode(scene, name)
{
}
