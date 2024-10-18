/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMaterial.cpp

#include "VrmlNodeMaterial.h"

#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeMaterial::initFields(VrmlNodeMaterial *node, VrmlNodeType *t)
{
    VrmlNodeTemplate::initFieldsHelper(node, t,
                                       exposedField("ambientIntensity", node->d_ambientIntensity),
                                       exposedField("diffuseColor", node->d_diffuseColor),
                                       exposedField("emissiveColor", node->d_emissiveColor),
                                       exposedField("shininess", node->d_shininess),
                                       exposedField("specularColor", node->d_specularColor),
                                       exposedField("transparency", node->d_transparency));

}

const char *VrmlNodeMaterial::name() { return "Material"; }

VrmlNodeMaterial::VrmlNodeMaterial(VrmlScene *scene)
    : VrmlNodeTemplate(scene, name())
    , d_ambientIntensity(0.2f)
    , d_diffuseColor(0.8f, 0.8f, 0.8f)
    , d_emissiveColor(0.0f, 0.0f, 0.0f)
    , d_shininess(0.2f)
    , d_specularColor(0.0f, 0.0f, 0.0f)
    , d_transparency(0.0f)
{
}

VrmlNodeMaterial *VrmlNodeMaterial::toMaterial() const
{
    return (VrmlNodeMaterial *)this;
}

// This currently isn't used - see VrmlNodeAppearance.cpp.

void VrmlNodeMaterial::render(Viewer *viewer)
{
    viewer->setMaterial(d_ambientIntensity.get(),
                        d_diffuseColor.get(),
                        d_emissiveColor.get(),
                        d_shininess.get(),
                        d_specularColor.get(),
                        d_transparency.get());
    clearModified();
}
