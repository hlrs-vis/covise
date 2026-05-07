/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeSky.cpp

#include "VrmlNodeSky.h"

#include "Sky.h"

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSky(scene);
}

VrmlNodeSky::VrmlNodeSky(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
{
}

VrmlNodeSky::VrmlNodeSky(const VrmlNodeSky &n)
    : VrmlNodeChild(n)
{
}

void VrmlNodeSky::initFields(VrmlNodeSky *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        field("skyName", node->d_skyName, [node](auto f)
            { Sky::instance()->setSky(node->d_skyName.get()); }),
        field("top", node->d_top, [node](auto f)
            { Sky::instance()->setTop(node->d_top.get()); }),
        field("bottom", node->d_bottom, [node](auto f)
            { Sky::instance()->setBottom(node->d_bottom.get()); }),
        field("floorColor", node->d_floorColor, [node](auto f)
            { Sky::instance()->setFloorColor(osg::Vec4(node->d_floorColor.get()[0], node->d_floorColor.get()[1], node->d_floorColor.get()[2], node->d_floorColor.get()[3])); }));
}

const char *VrmlNodeSky::typeName()
{
    return "Sky";
}
