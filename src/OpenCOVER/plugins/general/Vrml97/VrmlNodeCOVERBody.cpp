/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCOVERBody.cpp
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

#include "VrmlNodeCOVERBody.h"
#include "ViewerOsg.h"
#include <osg/Quat>

static list<VrmlNodeCOVERBody *> allCOVERBody;

void VrmlNodeCOVERBody::update()
{
    list<VrmlNodeCOVERBody *>::iterator ts;
    for (ts = allCOVERBody.begin(); ts != allCOVERBody.end(); ++ts)
    {
    }
}

void VrmlNodeCOVERBody::initFields(VrmlNodeCOVERBody *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("position", node->d_position),
                     exposedField("orientation", node->d_orientation),
                     exposedField("name", node->d_name, [node](auto f){
                            node->body=Input::instance()->getBody(node->d_name.get());
                     }),
                     exposedField("vrmlCoordinates", node->d_vrmlCoordinates));
}

const char *VrmlNodeCOVERBody::name()
{
    return "COVERBody";
}

VrmlNodeCOVERBody::VrmlNodeCOVERBody(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_position(0)
    , d_orientation(0.0)
    , d_name("noname")
    , d_vrmlCoordinates(true)
{
    body=Input::instance()->getBody(d_name.get());
    setModified();
}

void VrmlNodeCOVERBody::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        allCOVERBody.push_front(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeCOVERBody::VrmlNodeCOVERBody(const VrmlNodeCOVERBody &n)
    : VrmlNodeChild(n)
    , d_position(n.d_position)
    , d_orientation(n.d_orientation)
    , d_name(n.d_name)
    , d_vrmlCoordinates(n.d_vrmlCoordinates)
{
    body=Input::instance()->getBody(d_name.get());
    setModified();
}

VrmlNodeCOVERBody::~VrmlNodeCOVERBody()
{
    allCOVERBody.remove(this);
}

VrmlNodeCOVERBody *VrmlNodeCOVERBody::toCOVERBody() const
{
    return (VrmlNodeCOVERBody *)this;
}

void VrmlNodeCOVERBody::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();
    if(body)
    {
        osg::Matrix m;
        m = body->getMat();
	//fprintf(stderr,"bodyPos: %f %f %f %f\n",m(3,0),m(3,1),m(3,2),m(3,3));
	/*fprintf(stderr,"bodyPos: %f %f %f %f\n",m(0,0),m(0,1),m(0,2),m(0,3));
	fprintf(stderr,"bodyPos: %f %f %f %f\n",m(1,0),m(1,1),m(1,2),m(1,3));
	fprintf(stderr,"bodyPos: %f %f %f %f\n",m(2,0),m(2,1),m(2,2),m(2,3));
	fprintf(stderr,"bodyPos: %f %f %f %f\n",m(3,0),m(3,1),m(3,2),m(3,3));*/
        if(d_vrmlCoordinates.get())
        {
            osg::Matrix invbase = cover->getInvBaseMat();
            osg::Matrix VRMLRootMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
            osg::Matrix invVRMLRootMat;
            invVRMLRootMat.invert(VRMLRootMat);
            m = VRMLRootMat * m*invbase * invVRMLRootMat;
	    //remove scale from rotation part
	    osg::Vec3d v;
	    v.set(m(0,0),m(0,1),m(0,2));
	    v.normalize();
	    m(0,0)=v[0];m(0,1)=v[1];m(0,2)=v[2];
	    v.set(m(1,0),m(1,1),m(1,2));
	    v.normalize();
	    m(1,0)=v[0];m(1,1)=v[1];m(1,2)=v[2];
	    v.set(m(2,0),m(2,1),m(2,2));
	    v.normalize();
	    m(2,0)=v[0];m(2,1)=v[1];m(2,2)=v[2];
        }

        d_position.set(m.getTrans()[0], m.getTrans()[1], m.getTrans()[2]);
        eventOut(timeNow, "position", d_position);
        osg::Quat q;
        q.set(m);
        osg::Quat::value_type orient[4];
        q.getRotate(orient[3], orient[0], orient[1], orient[2]);
	//fprintf(stderr,"bodyOri: %f %f %f %f\n",orient[3], orient[0], orient[1], orient[2]);
        d_orientation.set(orient[0], orient[1], orient[2], orient[3]);

        eventOut(timeNow, "orientation", d_orientation);
    }

    setModified();
}
