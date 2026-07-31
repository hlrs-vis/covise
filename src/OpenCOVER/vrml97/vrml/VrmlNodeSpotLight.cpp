/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSpotLight.cpp

#include "VrmlNodeSpotLight.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Viewer.h"

using namespace vrml;

// Return a new VrmlNodeSpotLight
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeSpotLight(s); }

// Define the built in VrmlNodeType:: "SpotLight" fields

void VrmlNodeSpotLight::initFields(VrmlNodeSpotLight *node, VrmlNodeType *t)
{
    VrmlNodeLight::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("attenuation", &VrmlNodeSpotLight::d_attenuation),
                     exposedField("beamWidth", &VrmlNodeSpotLight::d_beamWidth),
                     exposedField("cutOffAngle", &VrmlNodeSpotLight::d_cutOffAngle),
                     exposedField("direction", &VrmlNodeSpotLight::d_direction),
                     exposedField("location", &VrmlNodeSpotLight::d_location),
                     exposedField("radius", &VrmlNodeSpotLight::d_radius));
}

const char *VrmlNodeSpotLight::typeName() { return "SpotLight"; }


VrmlNodeSpotLight::VrmlNodeSpotLight(VrmlScene *scene)
    : VrmlNodeLight(scene, typeName())
    , d_attenuation(1.0f, 0.0f, 0.0f)
    , d_beamWidth(1.570796f)
    , d_cutOffAngle(0.785398f)
    , d_direction(0.0f, 0.0f, -1.0f)
    , d_location(0.0f, 0.0f, 0.0f)
    , d_radius(100)
{
    setModified();
    if (d_scene)
        d_scene->addScopedLight(this);
}

VrmlNodeSpotLight::~VrmlNodeSpotLight()
{
    if (d_scene)
        d_scene->removeScopedLight(this);
}

void VrmlNodeSpotLight::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addScopedLight(this);
}

// This should be called before rendering any geometry in the scene.
// Since this is called from Scene::render() before traversing the
// scene graph, the proper transformation matrix hasn't been set up.
// Somehow it needs to figure out the accumulated xforms of its
// parents and apply them before rendering. This is not easy with
// DEF/USEd nodes...
void VrmlNodeSpotLight::render(Viewer *viewer)
{
    viewer->beginObject(name(), 0, this);
    if (isModified())
    {
        if (d_on.get() && d_radius.get() > 0.0)
            d_viewerObject = viewer->insertSpotLight(d_ambientIntensity.get(),
                                    d_attenuation.get(),
                                    d_beamWidth.get(),
                                    d_color.get(),
                                    d_cutOffAngle.get(),
                                    d_direction.get(),
                                    d_intensity.get(),
                                    d_location.get(),
                                    d_radius.get());
        else
            d_viewerObject = viewer->insertSpotLight(d_ambientIntensity.get(),
                                    d_attenuation.get(),
                                    d_beamWidth.get(),
                                    d_color.get(),
                                    d_cutOffAngle.get(),
                                    d_direction.get(),
                                    0.0,
                                    d_location.get(),
                                    d_radius.get());
        clearModified();
    }
    viewer->endObject();
}


// LarryD Mar 04/99
const VrmlSFVec3f &VrmlNodeSpotLight::getAttenuation() const
{
    return d_attenuation;
}

// LarryD Mar 04/99
const VrmlSFVec3f &VrmlNodeSpotLight::getDirection() const
{
    return d_direction;
}

// LarryD Mar 04/99
const VrmlSFVec3f &VrmlNodeSpotLight::getLocation() const
{
    return d_location;
}
