/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeDirLight.cpp

#include "VrmlNodeDirLight.h"
#include "VrmlNodeType.h"

#include "MathUtils.h"
#include "Viewer.h"

using namespace vrml;

// Return a new VrmlNodeDirLight
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeDirLight(s); }

// Define the built in VrmlNodeType:: "DirLight" fields

void VrmlNodeDirLight::initFields(VrmlNodeDirLight *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("direction", node->d_direction));
    VrmlNodeLight::initFields(node, t);
}

const char *VrmlNodeDirLight::name() { return "DirectionalLight"; }

VrmlNodeDirLight::VrmlNodeDirLight(VrmlScene *scene)
    : VrmlNodeLight(scene, name())
    , d_direction(0.0, 0.0, -1.0)
{
    setModified();
}

// This should be called before rendering any sibling nodes.

void VrmlNodeDirLight::render(Viewer *viewer)
{
    viewer->beginObject(name(), 0, this);
    if (isModified())
    {
        if (d_on.get())
            d_viewerObject = viewer->insertDirLight(d_ambientIntensity.get(),
                                   d_intensity.get(),
                                   d_color.get(),
                                   d_direction.get());
        else
            d_viewerObject = viewer->insertDirLight(d_ambientIntensity.get(),
                                   0.0,
                                   d_color.get(),
                                   d_direction.get());
        clearModified();
    }
    viewer->endObject();
}

// LarryD Mar 04/99
const VrmlSFVec3f &VrmlNodeDirLight::getDirection() const
{
    return d_direction;
}

// LarryD Mar 04/99
VrmlNodeDirLight *VrmlNodeDirLight::toDirLight() const
{
    return (VrmlNodeDirLight *)this;
}
