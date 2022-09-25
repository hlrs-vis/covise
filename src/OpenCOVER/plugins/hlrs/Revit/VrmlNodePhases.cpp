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

// Timesteps factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodePhases(scene);
}

void VrmlNodePhases::update()
{
    
}

// Define the built in VrmlNodeType:: "Timesteps" fields

VrmlNodeType *VrmlNodePhases::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Phases", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("numPhases", VrmlField::SFINT32);
    t->addExposedField("phase", VrmlField::SFINT32);
    t->addExposedField("phaseName", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodePhases::nodeType() const
{
    return defineType(0);
}

VrmlNodePhases::VrmlNodePhases(VrmlScene *scene)
    : VrmlNodeChild(scene)
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
    : VrmlNodeChild(n.d_scene)
    , d_numPhases(n.d_numPhases)
    , d_Phase(n.d_Phase)
    , d_PhaseName(n.d_PhaseName)
{
    theInstance = this;
    setModified();
}

VrmlNodePhases::~VrmlNodePhases()
{
}

VrmlNode *VrmlNodePhases::cloneMe() const
{
    return new VrmlNodePhases(*this);
}


void VrmlNodePhases::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();

    if (d_Phase.get() >0)
    {
        eventOut(timeNow, "phase", d_Phase);
        eventOut(timeNow, "phaseName", d_PhaseName);
    }
    if (d_numPhases.get() >0)
    {
        eventOut(timeNow, "numPhases", d_numPhases);
    }
}

ostream &VrmlNodePhases::printFields(ostream &os, int indent)
{
    if (!d_numPhases.get())
        PRINT_FIELD(numPhases);
    if (!d_Phase.get())
        PRINT_FIELD(Phase);
    if (!d_PhaseName.get())
        PRINT_FIELD(PhaseName);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodePhases::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(numPhases, SFInt)
    else if
        TRY_FIELD(Phase, SFInt)
    else if
        TRY_FIELD(PhaseName, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "Phase") == 0)
    {
        RevitPlugin::instance()->setPhase(d_Phase.get());
    }
    if (strcmp(fieldName, "Phase") == 0)
    {
        RevitPlugin::instance()->setPhase(d_Phase.get());
    }
}

const VrmlField *VrmlNodePhases::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "numPhases") == 0)
        return &d_numPhases;
    if (strcmp(fieldName, "phase") == 0)
        return &d_Phase;
    if (strcmp(fieldName, "phaseName") == 0)
        return &d_PhaseName;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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