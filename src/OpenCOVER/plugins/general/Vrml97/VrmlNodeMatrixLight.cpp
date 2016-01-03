/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeMatrixLight.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>

#include "VrmlNodeMatrixLight.h"
#include "ViewerOsg.h"
#include <osg/Quat>

static list<VrmlNodeMatrixLight *> allMatrixLights;

// MatrixLight factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeMatrixLight(scene);
}

void VrmlNodeMatrixLight::update()
{
}

// Define the built in VrmlNodeType:: "MatrixLight" fields

VrmlNodeType *VrmlNodeMatrixLight::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("MatrixLight", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("numMatrixLight", VrmlField::SFINT32);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("loop", VrmlField::SFBOOL);
    t->addEventOut("fraction_changed", VrmlField::SFFLOAT);
    t->addEventIn("timestep", VrmlField::SFINT32);

    return t;
}

VrmlNodeType *VrmlNodeMatrixLight::nodeType() const
{
    return defineType(0);
}

VrmlNodeMatrixLight::VrmlNodeMatrixLight(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_numMatrixLight(0)
    , d_fraction_changed(0.0)
    , d_enabled(true)
    , d_loop(true)
{
    setModified();
    precipitationEffect = new coMatrixLightEffect;
    precipitationEffect->rain(0.5);
    //precipitationEffect->setParticleSize(0.03*1000);
    cover->getObjectsRoot()->addChild(precipitationEffect.get());
}

void VrmlNodeMatrixLight::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeMatrixLight::VrmlNodeMatrixLight(const VrmlNodeMatrixLight &n)
    : VrmlNodeChild(n.d_scene)
    , d_numMatrixLight(n.d_numMatrixLight)
    , d_fraction_changed(n.d_fraction_changed)
    , d_enabled(n.d_enabled)
    , d_loop(n.d_loop)
{
    
    precipitationEffect = new coMatrixLightEffect;
    precipitationEffect->rain(0.5);
   // precipitationEffect->setNearTransition(100);
    //precipitationEffect->setFarTransition(100000);
    //precipitationEffect->setParticleSize(0.03*1000);
    
    cover->getObjectsRoot()->addChild(precipitationEffect.get());
    setModified();
}

VrmlNodeMatrixLight::~VrmlNodeMatrixLight()
{
    cover->getObjectsRoot()->removeChild(precipitationEffect.get());
}

VrmlNode *VrmlNodeMatrixLight::cloneMe() const
{
    return new VrmlNodeMatrixLight(*this);
}

VrmlNodeMatrixLight *VrmlNodeMatrixLight::toMatrixLight() const
{
    return (VrmlNodeMatrixLight *)this;
}

void VrmlNodeMatrixLight::render(Viewer *viewer)
{
    (void)viewer;
    //setModified();
}

ostream &VrmlNodeMatrixLight::printFields(ostream &os, int indent)
{
    if (!d_numMatrixLight.get())
        PRINT_FIELD(numMatrixLight);
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_loop.get())
        PRINT_FIELD(loop);
    if (!d_fraction_changed.get())
        PRINT_FIELD(fraction_changed);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMatrixLight::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(numMatrixLight, SFInt)
    else if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(loop, SFBool)
    else if
        TRY_FIELD(fraction_changed, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "numMatrixLight") == 0)
    {
    }
    if (strcmp(fieldName, "timestep") == 0)
    {
    }
}

const VrmlField *VrmlNodeMatrixLight::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "numMatrixLight") == 0)
        return &d_numMatrixLight;
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "loop") == 0)
        return &d_loop;
    else if (strcmp(fieldName, "fraction_changed") == 0)
        return &d_fraction_changed;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
