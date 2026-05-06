/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeSkySphere.cpp

#include "VrmlNodeSkySphere.h"

#include "SkySphere.h"

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSkySphere(scene);
}

VrmlNodeSkySphere::VrmlNodeSkySphere(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
{
}

VrmlNodeSkySphere::VrmlNodeSkySphere(const VrmlNodeSkySphere &n)
    : VrmlNodeChild(n)
{
}

void VrmlNodeSkySphere::initFields(VrmlNodeSkySphere *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        field("skyName", node->d_skyName, [node](auto f)
            { SkySphere::instance()->setSky(node->d_skyName.get()); }),
        field("top", node->d_top, [node](auto f)
            { SkySphere::instance()->setTop(node->d_top.get()); }),
        field("bottom", node->d_bottom, [node](auto f)
            { SkySphere::instance()->setBottom(node->d_bottom.get()); }),
        field("floorColor", node->d_floorColor, [node](auto f)
            { SkySphere::instance()->setFloorColor(osg::Vec4(node->d_floorColor.get()[0], node->d_floorColor.get()[1], node->d_floorColor.get()[2], node->d_floorColor.get()[3])); }));
}

const char *VrmlNodeSkySphere::typeName()
{
    return "SkySphere";
}
