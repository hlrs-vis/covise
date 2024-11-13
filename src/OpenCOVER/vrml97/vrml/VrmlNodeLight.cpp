/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeLight.cpp

#include "VrmlNodeLight.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"

using namespace vrml;

// Define the built in VrmlNodeType:: "Light" fields

void VrmlNodeLight::initFields(VrmlNodeLight *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("ambientIntensity", node->d_ambientIntensity),
                     exposedField("color", node->d_color),
                     exposedField("intensity", node->d_intensity),
                     exposedField("on", node->d_on));
    VrmlNodeChild::initFields(node, t);                     
}

VrmlNodeLight::VrmlNodeLight(VrmlScene *scene, const std::string &name)
    : VrmlNodeChild(scene, name)
    , d_ambientIntensity(0.0)
    , d_color(1.0, 1.0, 1.0)
    , d_intensity(1.0)
    , d_on(true)
{
}

void VrmlNodeLight::render(Viewer *)
{
}
