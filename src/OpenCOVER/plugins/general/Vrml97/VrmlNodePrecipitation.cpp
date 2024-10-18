/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePrecipitation.cpp
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

#include "VrmlNodePrecipitation.h"
#include "ViewerOsg.h"
#include <osg/Quat>

static list<VrmlNodePrecipitation *> allPrecipitation;

void VrmlNodePrecipitation::initFields(VrmlNodePrecipitation *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("numPrecipitation", node->d_numPrecipitation),
                     exposedField("enabled", node->d_enabled),
                     exposedField("loop", node->d_loop));
    if (t)
    {
        t->addEventOut("fraction_changed", VrmlField::SFFLOAT);
        t->addEventIn("timestep", VrmlField::SFINT32);
    }
}

const char *VrmlNodePrecipitation::name()
{
    return "Precipitation";
}

VrmlNodePrecipitation::VrmlNodePrecipitation(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_numPrecipitation(0)
    , d_fraction_changed(0.0)
    , d_enabled(true)
    , d_loop(true)
{
    setModified();
    precipitationEffect = new coPrecipitationEffect;
    precipitationEffect->rain(0.5);
    //precipitationEffect->setParticleSize(0.03*1000);
    cover->getObjectsRoot()->addChild(precipitationEffect.get());
}

void VrmlNodePrecipitation::addToScene(VrmlScene *s, const char *relUrl)
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

VrmlNodePrecipitation::VrmlNodePrecipitation(const VrmlNodePrecipitation &n)
    : VrmlNodeChild(n)
    , d_numPrecipitation(n.d_numPrecipitation)
    , d_fraction_changed(n.d_fraction_changed)
    , d_enabled(n.d_enabled)
    , d_loop(n.d_loop)
{
    
    precipitationEffect = new coPrecipitationEffect;
    precipitationEffect->rain(0.5);
   // precipitationEffect->setNearTransition(100);
    //precipitationEffect->setFarTransition(100000);
    //precipitationEffect->setParticleSize(0.03*1000);
    
    cover->getObjectsRoot()->addChild(precipitationEffect.get());
    setModified();
}

VrmlNodePrecipitation::~VrmlNodePrecipitation()
{
    cover->getObjectsRoot()->removeChild(precipitationEffect.get());
}

VrmlNodePrecipitation *VrmlNodePrecipitation::toPrecipitation() const
{
    return (VrmlNodePrecipitation *)this;
}

void VrmlNodePrecipitation::render(Viewer *viewer)
{
    (void)viewer;
    //setModified();
}

const VrmlField *VrmlNodePrecipitation::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "fraction_changed") == 0)
        return &d_fraction_changed;
    else
        return VrmlNodeChild::getField(fieldName);
}
