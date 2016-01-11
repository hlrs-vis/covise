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

// Return a new VrmlNodePointLight
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodePointLight(s); }

// Define the built in VrmlNodeType:: "PointLight" fields

VrmlNodeType *VrmlNodePointLight::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("PointLight", creator);
    }

    VrmlNodeLight::defineType(t); // Parent class
    t->addExposedField("attenuation", VrmlField::SFVEC3F);
    t->addExposedField("location", VrmlField::SFVEC3F);
    t->addExposedField("radius", VrmlField::SFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodePointLight::nodeType() const { return defineType(0); }

VrmlNodePointLight::VrmlNodePointLight(VrmlScene *scene)
    : VrmlNodeLight(scene)
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

VrmlNode *VrmlNodePointLight::cloneMe() const
{
    return new VrmlNodePointLight(*this);
}

VrmlNodePointLight *VrmlNodePointLight::toPointLight() const
{
    return (VrmlNodePointLight *)this;
}

void VrmlNodePointLight::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addScopedLight(this);
}

std::ostream &VrmlNodePointLight::printFields(std::ostream &os, int indent)
{
    VrmlNodeLight::printFields(os, indent);
    if (!FPEQUAL(d_attenuation.x(), 1.0) || !FPZERO(d_attenuation.y()) || !FPZERO(d_attenuation.z()))
        PRINT_FIELD(attenuation);
    if (!FPZERO(d_location.x()) || !FPZERO(d_location.y()) || !FPZERO(d_location.z()))
        PRINT_FIELD(location);
    if (!FPEQUAL(d_radius.get(), 100.0))
        PRINT_FIELD(radius);

    return os;
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

// Set the value of one of the node fields.

void VrmlNodePointLight::setField(const char *fieldName,
                                  const VrmlField &fieldValue)
{
    if
        TRY_FIELD(attenuation, SFVec3f)
    else if
        TRY_FIELD(location, SFVec3f)
    else if
        TRY_FIELD(radius, SFFloat)
    else
        VrmlNodeLight::setField(fieldName, fieldValue);
    setModified();
}

const VrmlField *VrmlNodePointLight::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "attenuation") == 0)
        return &d_attenuation;
    else if (strcmp(fieldName, "location") == 0)
        return &d_location;
    else if (strcmp(fieldName, "radius") == 0)
        return &d_radius;

    return VrmlNodeLight::getField(fieldName);
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
