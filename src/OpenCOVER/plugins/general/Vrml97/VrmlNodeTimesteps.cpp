/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeTimesteps.cpp
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

#include "VrmlNodeTimesteps.h"
#include "ViewerOsg.h"
#include <osg/Quat>

static list<VrmlNodeTimesteps *> allTimesteps;

// Timesteps factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeTimesteps(scene);
}

void VrmlNodeTimesteps::update()
{
    list<VrmlNodeTimesteps *>::iterator ts;
    for (ts = allTimesteps.begin(); ts != allTimesteps.end(); ++ts)
    {
    }
}

void VrmlNodeTimesteps::initFields(VrmlNodeTimesteps *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t, 
                     field("numTimesteps", node->d_numTimesteps, [node](auto f){
                            coVRAnimationManager::instance()->setNumTimesteps(node->d_numTimesteps.get(), node);
                     }),
                     field("enabled", node->d_enabled, [](auto f){
                            coVRAnimationManager::instance()->enableAnimation(f->get());
                     }),
                     field("loop", node->d_loop),
                     field("maxFrameRate", node->d_maxFrameRate, [](auto f){
                            coVRAnimationManager::instance()->setMaxFrameRate(f->get());
                     }));
    
    if(t)
    {
        t->addEventOut("fraction_changed", VrmlField::SFFLOAT);
        t->addEventOut("timestep_changed", VrmlField::SFINT32);
        t->addEventIn("timestep", VrmlField::SFINT32); //event might not be handled
    }                     

}

const char *VrmlNodeTimesteps::name()
{
    return "Timesteps";
}

VrmlNodeTimesteps::VrmlNodeTimesteps(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_numTimesteps(0)
    , d_fraction_changed(0.0)
    , d_enabled(true)
    , d_loop(true)
    , d_maxFrameRate(0)
{
    coVRAnimationManager::instance()->showAnimMenu(true);
    setModified();
}

void VrmlNodeTimesteps::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        allTimesteps.push_front(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeTimesteps::VrmlNodeTimesteps(const VrmlNodeTimesteps &n)
    : VrmlNodeChild(n)
    , d_numTimesteps(n.d_numTimesteps)
    , d_fraction_changed(n.d_fraction_changed)
    , d_enabled(n.d_enabled)
    , d_loop(n.d_loop)
    , d_maxFrameRate(n.d_maxFrameRate)
{
    coVRAnimationManager::instance()->showAnimMenu(true);
    setModified();
}

VrmlNodeTimesteps::~VrmlNodeTimesteps()
{
    allTimesteps.remove(this);
}

VrmlNodeTimesteps *VrmlNodeTimesteps::toTimesteps() const
{
    return (VrmlNodeTimesteps *)this;
}

void VrmlNodeTimesteps::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();
    int numTimeSteps = d_numTimesteps.get();

    float fraction = 0;
    if(numTimeSteps<=0)
    {
        numTimeSteps = coVRAnimationManager::instance()->getNumTimesteps();
    }
    else
    {
        fraction = float(coVRAnimationManager::instance()->getAnimationFrame() % numTimeSteps) / (float)numTimeSteps;
    }
    if (d_fraction_changed.get() != fraction)
    {
        d_fraction_changed.set(fraction);
        eventOut(timeNow, "fraction_changed", d_fraction_changed);
    }
    if (d_numTimesteps.get() <=0 && d_numTimesteps.get() != coVRAnimationManager::instance()->getNumTimesteps())
    {
        d_numTimesteps.set(coVRAnimationManager::instance()->getNumTimesteps());
        eventOut(timeNow, "numTimesteps", d_numTimesteps);
    }
    if(d_currentTimestep.get() != coVRAnimationManager::instance()->getAnimationFrame())
    {
        d_currentTimestep.set(coVRAnimationManager::instance()->getAnimationFrame());
        eventOut(timeNow, "timestep_changed", d_currentTimestep);
    }
    setModified();
}
