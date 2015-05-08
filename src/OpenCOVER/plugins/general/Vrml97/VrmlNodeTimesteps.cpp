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

// Define the built in VrmlNodeType:: "Timesteps" fields

VrmlNodeType *VrmlNodeTimesteps::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Timesteps", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("numTimesteps", VrmlField::SFINT32);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("loop", VrmlField::SFBOOL);
    t->addEventOut("fraction_changed", VrmlField::SFFLOAT);
    t->addEventIn("timestep", VrmlField::SFINT32);

    return t;
}

VrmlNodeType *VrmlNodeTimesteps::nodeType() const
{
    return defineType(0);
}

VrmlNodeTimesteps::VrmlNodeTimesteps(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_numTimesteps(0)
    , d_fraction_changed(0.0)
    , d_enabled(true)
    , d_loop(true)
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
    : VrmlNodeChild(n.d_scene)
    , d_numTimesteps(n.d_numTimesteps)
    , d_fraction_changed(n.d_fraction_changed)
    , d_enabled(n.d_enabled)
    , d_loop(n.d_loop)
{
    coVRAnimationManager::instance()->showAnimMenu(true);
    setModified();
}

VrmlNodeTimesteps::~VrmlNodeTimesteps()
{
    allTimesteps.remove(this);
}

VrmlNode *VrmlNodeTimesteps::cloneMe() const
{
    return new VrmlNodeTimesteps(*this);
}

VrmlNodeTimesteps *VrmlNodeTimesteps::toTimesteps() const
{
    return (VrmlNodeTimesteps *)this;
}

void VrmlNodeTimesteps::render(Viewer *viewer)
{
    (void)viewer;
    double timeNow = System::the->time();
    if (d_fraction_changed.get() != (float)coVRAnimationManager::instance()->getAnimationFrame() / (float)coVRAnimationManager::instance()->getNumTimesteps())
    {
        d_fraction_changed.set((float)coVRAnimationManager::instance()->getAnimationFrame() / (float)coVRAnimationManager::instance()->getNumTimesteps());
        eventOut(timeNow, "fraction_changed", d_fraction_changed);
    }
    if (d_numTimesteps.get() != coVRAnimationManager::instance()->getNumTimesteps())
    {
        d_numTimesteps.set(coVRAnimationManager::instance()->getNumTimesteps());
        eventOut(timeNow, "numTimesteps", d_numTimesteps);
    }
    setModified();
}

ostream &VrmlNodeTimesteps::printFields(ostream &os, int indent)
{
    if (!d_numTimesteps.get())
        PRINT_FIELD(numTimesteps);
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_loop.get())
        PRINT_FIELD(loop);
    if (!d_fraction_changed.get())
        PRINT_FIELD(fraction_changed);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeTimesteps::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(numTimesteps, SFInt)
    else if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(loop, SFBool)
    else if
        TRY_FIELD(fraction_changed, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "numTimesteps") == 0)
    {
        coVRAnimationManager::instance()->setNumTimesteps(d_numTimesteps.get(), this);
    }
    if (strcmp(fieldName, "timestep") == 0)
    {
        VrmlSFInt frame = (VrmlSFInt &)fieldValue;
        coVRAnimationManager::instance()->requestAnimationFrame(frame.get());
    }
}

const VrmlField *VrmlNodeTimesteps::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "numTimesteps") == 0)
        return &d_numTimesteps;
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "loop") == 0)
        return &d_loop;
    else if (strcmp(fieldName, "fraction_changed") == 0)
        return &d_fraction_changed;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
