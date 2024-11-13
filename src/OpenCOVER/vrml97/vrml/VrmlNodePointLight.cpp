/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePointLight.cpp

#include "VrmlNodePointLight.h"

#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodePointLight::initFields(VrmlNodePointLight *node, VrmlNodeType *t)
{
    VrmlNodeLight::initFields(node, t); 
    initFieldsHelper(node, t,
        exposedField("attenuation", node->d_attenuation),
        exposedField("location", node->d_location),
        exposedField("radius", node->d_radius)
    );
}

const char *VrmlNodePointLight::name() { return "PointLight"; }


VrmlNodePointLight::VrmlNodePointLight(VrmlScene *scene)
    : VrmlNodeLight(scene, name())
    , d_attenuation(1.0, 0.0, 0.0)
    , d_location(0.0, 0.0, 0.0)
    , d_radius(100)
{
    setModified();
    if (d_scene)
        d_scene->addScopedLight(this);
}

VrmlNodePointLight::~VrmlNodePointLight()
{
    if (d_scene)
        d_scene->removeScopedLight(this);
}

void VrmlNodePointLight::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addScopedLight(this);
}

// This should be called before rendering any geometry nodes in the scene.
// Since this is called from Scene::render() before traversing the
// scene graph, the proper transformation matrix hasn't been set up.
// Somehow it needs to figure out the accumulated xforms of its
// parents and apply them before rendering. This is not easy with
// DEF/USEd nodes...

void VrmlNodePointLight::render(Viewer *viewer)
{
    viewer->beginObject(name(), 0, this);
    if (isModified())
    {
        if (d_on.get() && d_radius.get() > 0.0)
            d_viewerObject = viewer->insertPointLight(d_ambientIntensity.get(),
                                     d_attenuation.get(),
                                     d_color.get(),
                                     d_intensity.get(),
                                     d_location.get(),
                                     d_radius.get());
        else
            d_viewerObject = viewer->insertPointLight(d_ambientIntensity.get(),
                                     d_attenuation.get(),
                                     d_color.get(),
                                     0.0,
                                     d_location.get(),
                                     d_radius.get());
        clearModified();
    }
    viewer->endObject();
}

// LarryD Mar 04/99
const VrmlSFVec3f &VrmlNodePointLight::getAttenuation() const
{
    return d_attenuation;
}

// LarryD Mar 04/99
const VrmlSFVec3f &VrmlNodePointLight::getLocation() const
{
    return d_location;
}
