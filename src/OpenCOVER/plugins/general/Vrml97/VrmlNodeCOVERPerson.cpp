/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCOVERPerson.cpp
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
#include <cover/input/input.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>

#include "VrmlNodeCOVERPerson.h"
#include "ViewerOsg.h"
#include <osg/Quat>

static list<VrmlNodeCOVERPerson *> allCOVERPerson;

// COVERPerson factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeCOVERPerson(scene);
}

void VrmlNodeCOVERPerson::update()
{
    list<VrmlNodeCOVERPerson *>::iterator ts;
    for (ts = allCOVERPerson.begin(); ts != allCOVERPerson.end(); ++ts)
    {
    }
}

// Define the built in VrmlNodeType:: "COVERPerson" fields

VrmlNodeType *VrmlNodeCOVERPerson::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("COVERPerson", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("activePerson", VrmlField::SFINT32);
    t->addExposedField("eyeDistance", VrmlField::SFFLOAT);
    return t;
}

VrmlNodeType *VrmlNodeCOVERPerson::nodeType() const
{
    return defineType(0);
}

VrmlNodeCOVERPerson::VrmlNodeCOVERPerson(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_activePerson(Input::instance()->getActivePerson())
{
    setModified();
}

void VrmlNodeCOVERPerson::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        allCOVERPerson.push_front(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeCOVERPerson::VrmlNodeCOVERPerson(const VrmlNodeCOVERPerson &n)
    : VrmlNodeChild(n.d_scene)
    , d_activePerson(n.d_activePerson)
{
    setModified();
}

VrmlNodeCOVERPerson::~VrmlNodeCOVERPerson()
{
    allCOVERPerson.remove(this);
}

VrmlNode *VrmlNodeCOVERPerson::cloneMe() const
{
    return new VrmlNodeCOVERPerson(*this);
}

VrmlNodeCOVERPerson *VrmlNodeCOVERPerson::toCOVERPerson() const
{
    return (VrmlNodeCOVERPerson *)this;
}

void VrmlNodeCOVERPerson::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();
    if (d_activePerson.get() != Input::instance()->getActivePerson())
    {
        d_activePerson.set(Input::instance()->getActivePerson());
        eventOut(timeNow, "activePerson", d_activePerson);
    }
    setModified();
}

ostream &VrmlNodeCOVERPerson::printFields(ostream &os, int indent)
{
    if (!d_activePerson.get())
        PRINT_FIELD(activePerson);
    if (!d_eyeDistance.get())
        PRINT_FIELD(eyeDistance);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCOVERPerson::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(activePerson, SFInt)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "activePerson") == 0)
    {
        Input::instance()->setActivePerson(d_activePerson.get());
    }
    else if (strcmp(fieldName, "eyeDistance") == 0)
    {
        VRViewer::instance()->setSeparation(d_eyeDistance.get());
    }
}

const VrmlField *VrmlNodeCOVERPerson::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "activePerson") == 0)
        return &d_activePerson;
    else if (strcmp(fieldName, "eyeDistance") == 0)
        return &d_eyeDistance;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
