/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePhases.cpp

#define QT_NO_EMIT
#include "RevitPlugin.h"
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


#include "VrmlNodePhases.h"

VrmlNodePhases *VrmlNodePhases::theInstance = nullptr;

void VrmlNodePhases::update()
{
    
}

void VrmlNodePhases::initFields(VrmlNodePhases *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("numPhases", node->d_numPhases),
                     exposedField("phase", node->d_Phase, [](auto f){
                         RevitPlugin::instance()->setPhase(f->get());
                     }),
                     exposedField("phaseName", node->d_PhaseName));
}

const char *VrmlNodePhases::name() { return "Phases"; }

VrmlNodePhases::VrmlNodePhases(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_numPhases(0)
    , d_Phase(-1)
    , d_PhaseName("")
{
    theInstance = this;
    setModified();
}

void VrmlNodePhases::addToScene(VrmlScene *s, const char *relUrl)
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

VrmlNodePhases::VrmlNodePhases(const VrmlNodePhases &n)
    : VrmlNodeChild(n)
    , d_numPhases(n.d_numPhases)
    , d_Phase(n.d_Phase)
    , d_PhaseName(n.d_PhaseName)
{
    theInstance = this;
    setModified();
}

void VrmlNodePhases::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();

    /*if (d_Phase.get() >= 0)
    {
        eventOut(timeNow, "phase", d_Phase);
        eventOut(timeNow, "phaseName", d_PhaseName);
    }
    if (d_numPhases.get() >0)
    {
        eventOut(timeNow, "numPhases", d_numPhases);
    }*/
}

void VrmlNodePhases::setPhase(int phase)
{
    d_Phase = phase;
    double timeNow = System::the->time();
    eventOut(timeNow, "phase", d_Phase);

}
void VrmlNodePhases::setPhase(const std::string& phaseName)
{
    d_PhaseName = phaseName.c_str();
    double timeNow = System::the->time();
    eventOut(timeNow, "phaseName", d_PhaseName);
}

void VrmlNodePhases::setNumPhases(int numphases)
{
    d_numPhases = numphases;
    double timeNow = System::the->time();
    eventOut(timeNow, "numPhases", d_numPhases);
}