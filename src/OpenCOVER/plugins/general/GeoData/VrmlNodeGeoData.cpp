/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeGeoData.cpp
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

#include "GeoDataLoader.h"
#include "VrmlNodeGeoData.h"
#include <osg/Quat>

static list<VrmlNodeGeoData *> allGeoData;

// GeoData factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeGeoData(scene);
}

void VrmlNodeGeoData::update()
{
    list<VrmlNodeGeoData *>::iterator ts;
    for (ts = allGeoData.begin(); ts != allGeoData.end(); ++ts)
    {
    }
}

void VrmlNodeGeoData::initFields(VrmlNodeGeoData *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t, 
                     field("offset", node->d_offset, [node](auto f){
                            GeoDataLoader::instance()->setOffset(osg::Vec3(node->d_offset.get()[0], node->d_offset.get()[1], node->d_offset.get()[2]));
                     }),
        field("skyName", node->d_skyName, [node](auto f) {
            GeoDataLoader::instance()->setSky(node->d_skyName.get());
            }),
        field("enabled", node->d_enabled, [node](auto f) {
            }));
                   

}

const char *VrmlNodeGeoData::typeName() 
{
    return "GeoData";
}

VrmlNodeGeoData::VrmlNodeGeoData(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
    , d_offset(0,0,0)
    , d_enabled(true)
    , d_skyName("")
{
    coVRAnimationManager::instance()->showAnimMenu(true);
    setModified();
}

void VrmlNodeGeoData::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        allGeoData.push_front(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeGeoData::VrmlNodeGeoData(const VrmlNodeGeoData &n)
    : VrmlNodeChild(n)
    , d_offset(0, 0, 0)
    , d_enabled(true)
    , d_skyName("")
{
    setModified();
}

VrmlNodeGeoData::~VrmlNodeGeoData()
{
    allGeoData.remove(this);
}

VrmlNodeGeoData *VrmlNodeGeoData::toGeoData() const
{
    return (VrmlNodeGeoData *)this;
}

void VrmlNodeGeoData::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();

}
