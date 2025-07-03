/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeJAKA.cpp
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
#include <math.h>
#include "VrmlNodeJAKA.h"
#include "JAKAPlugin.h"




void VrmlNodeJAKA::initFields(VrmlNodeJAKA* node, vrml::VrmlNodeType* t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("speed", node->d_speed, [node](auto f) {
            }),
        exposedField("lademodus", node->d_lademodus, [node](auto f) {
            }),
        eventOutCallBack("position", node->d_position, [node](auto f) {
            }),
        eventOutCallBack("rotation", node->d_rotation, [node](auto f) {
            }),
        eventOutCallBack("main0Angle", node->d_main0Angle, [node](auto f) {
            }),
        eventOutCallBack("main1Angle", node->d_main1Angle, [node](auto f) {
            }),
        eventOutCallBack("main2Angle", node->d_main2Angle, [node](auto f) {
            }),
        eventOutCallBack("sec0Angle", node->d_sec0Angle, [node](auto f) {
            }),
        eventOutCallBack("sec1Angle", node->d_sec1Angle, [node](auto f) {
            }),
        eventOutCallBack("sec2Angle", node->d_sec2Angle, [node](auto f) {
            })
        );
    if (t)
    {
        t->addEventIn("set_time", VrmlField::SFTIME);
    }
}

const char* VrmlNodeJAKA::typeName()
{
    return "JAKA";
}


VrmlNodeJAKA::VrmlNodeJAKA(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
    , d_speed(1.0)
    , d_lademodus(true)
    , d_position(0,0,0)
    , d_rotation(1,0,0,0)
    , d_main0Angle(0.0)
    , d_main1Angle(0.0)
    , d_main2Angle(0.0)
    , d_sec0Angle(0.0)
    , d_sec1Angle(0.0)
    , d_sec2Angle(0.0)
{
    setModified();
    

}

void VrmlNodeJAKA::addToScene(VrmlScene *s, const char *relUrl)
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

VrmlNodeJAKA::VrmlNodeJAKA(const VrmlNodeJAKA &n)
    : VrmlNodeChild(n)
    , d_speed(n.d_speed)
    , d_lademodus(n.d_lademodus)
    , d_position(n.d_position)
    , d_rotation(n.d_rotation)
    , d_main0Angle(n.d_main0Angle)
    , d_main1Angle(n.d_main1Angle)
    , d_main2Angle(n.d_main2Angle)
    , d_sec0Angle(n.d_sec0Angle)
    , d_sec1Angle(n.d_sec1Angle)
    , d_sec2Angle(n.d_sec2Angle)
{
    setModified();
}

VrmlNodeJAKA::~VrmlNodeJAKA()
{
}

void VrmlNodeJAKA::render(Viewer *viewer)
{
    (void)viewer;
    
    setModified();
}


