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

// COVERBody factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeCOVERBody(scene);
}

void VrmlNodeCOVERBody::update()
{
    list<VrmlNodeCOVERBody *>::iterator ts;
    for (ts = allCOVERBody.begin(); ts != allCOVERBody.end(); ++ts)
    {
    }
}

// Define the built in VrmlNodeType:: "COVERBody" fields

VrmlNodeType *VrmlNodeCOVERBody::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("COVERBody", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("position", VrmlField::SFVEC3F);
    t->addExposedField("orientation", VrmlField::SFROTATION);
    t->addExposedField("name", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeCOVERBody::nodeType() const
{
    return defineType(0);
}

VrmlNodeCOVERBody::VrmlNodeCOVERBody(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_position(0)
    , d_orientation(0.0)
    , d_name("noname")
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
    : VrmlNodeChild(n.d_scene)
    , d_position(n.d_position)
    , d_orientation(n.d_orientation)
    , d_name(n.d_name)
{
    body=Input::instance()->getBody(d_name.get());
    setModified();
}

VrmlNodeCOVERBody::~VrmlNodeCOVERBody()
{
    allCOVERBody.remove(this);
}

VrmlNode *VrmlNodeCOVERBody::cloneMe() const
{
    return new VrmlNodeCOVERBody(*this);
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
        d_position.set(m.getTrans()[0], m.getTrans()[1], m.getTrans()[2]);
        eventOut(timeNow, "position_changed", d_position);
        osg::Quat q;
        q.set(m);
        osg::Quat::value_type orient[4];
        q.getRotate(orient[3], orient[0], orient[1], orient[2]);
        d_orientation.set(orient[0], orient[1], orient[2], orient[3]);

        eventOut(timeNow, "orientation_changed", d_orientation);
    }

    setModified();
}

ostream &VrmlNodeCOVERBody::printFields(ostream &os, int indent)
{
    if (!d_position.get())
        PRINT_FIELD(position);
    if (!d_name.get())
        PRINT_FIELD(name);
    if (!d_orientation.get())
        PRINT_FIELD(orientation);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCOVERBody::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(position, SFVec3f)
    else if
        TRY_FIELD(name, SFString)
    else if
        TRY_FIELD(orientation, SFRotation)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
    if(strcmp(fieldName,"name")==0)
    {
        body=Input::instance()->getBody(d_name.get());
    }
}

const VrmlField *VrmlNodeCOVERBody::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "position") == 0)
        return &d_position;
    if (strcmp(fieldName, "name") == 0)
        return &d_name;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
