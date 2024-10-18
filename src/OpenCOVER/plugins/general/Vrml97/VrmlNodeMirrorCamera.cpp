/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeMirrorCamera.cpp
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
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>

#include "VrmlNodeMirrorCamera.h"
#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>

#define PI_OVER_180 0.0174532925199432957692369076849f
#define _180_OVER_PI 57.2957795130823208767981548141f

#define DEG_TO_RAD(x) (x * PI_OVER_180)
#define RAD_TO_DEG(x) (x * _180_OVER_PI)

static list<VrmlNodeMirrorCamera *> cameraNodes;

void VrmlNodeMirrorCamera::initFields(VrmlNodeMirrorCamera *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("hFov", node->d_hFov),
                     exposedField("vFov", node->d_vFov),
                     exposedField("far", node->d_far),
                     exposedField("near", node->d_near),
                     exposedField("enabled", node->d_enabled),
                     exposedField("cameraID", node->d_cameraID));
}

const char *VrmlNodeMirrorCamera::name()
{
    return "MirrorCamera";
}

VrmlNodeMirrorCamera::VrmlNodeMirrorCamera(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_hFov(90.0)
    , d_vFov(90.0)
    , d_near(0.1)
    , d_far(10000.0)
    , d_enabled(true)
    , d_cameraID(1)
{
}

void VrmlNodeMirrorCamera::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        addMC(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

VrmlNodeMirrorCamera::VrmlNodeMirrorCamera(const VrmlNodeMirrorCamera &n)
    : VrmlNodeChild(n)
    , d_hFov(n.d_hFov)
    , d_vFov(n.d_vFov)
    , d_near(n.d_near)
    , d_far(n.d_far)
    , d_enabled(n.d_enabled)
    , d_cameraID(n.d_cameraID)
{
    setModified();
}

VrmlNodeMirrorCamera::~VrmlNodeMirrorCamera()
{
    removeMC(this);
}

void VrmlNodeMirrorCamera::render(Viewer *v)
{
    ViewerOsg *viewer = (ViewerOsg *)v;

    double timeNow = System::the->time();
    for (int i = 0; i < viewer->numCameras; i++)
    {
        if (viewer->mirrors[i].CameraID == d_cameraID.get())
        {

            float nearC = d_near.get();
            float farC = d_far.get();

            float tangent = tanf(DEG_TO_RAD(d_vFov.get()) / 2.0f); // tangent of half vertical fov
            float height = nearC * tangent; // half height of near plane
            tangent = tanf(DEG_TO_RAD(d_hFov.get()) / 2.0f); // tangent of half vertical fov
            float width = nearC * tangent; // half width of near plane

            viewer->mirrors[i].pm = osg::Matrix::frustum(-width, width, -height, height, nearC, farC);

            osg::Matrix VRMLRootMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
            osg::Matrix tr;
            //VrmlMat4 currentTransform;
            tr = viewer->currentTransform;
            viewer->mirrors[i].vm =tr; // update vm one frame later
        }
    }

    setModified();
}

void VrmlNodeMirrorCamera::addMC(VrmlNodeMirrorCamera *node)
{
    cameraNodes.push_front(node);
}

void VrmlNodeMirrorCamera::removeMC(VrmlNodeMirrorCamera *node)
{
    cameraNodes.remove(node);
}
