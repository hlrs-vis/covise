/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeOffice.cpp
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
#include "VrmlNodeOffice.h"
#include "OfficePlugin.h"
#include <net/tokenbuffer.h>

list<VrmlNodeOffice *> VrmlNodeOffice::allOffice;

void VrmlNodeOffice::setMessage(const char *s)
{
    d_events.set(s);
    double timeNow = System::the->time();
    eventOut(timeNow, "events",d_events);
}


// Office factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeOffice(scene);
}

void VrmlNodeOffice::update()
{
    list<VrmlNodeOffice *>::iterator ts;
    for (ts = allOffice.begin(); ts != allOffice.end(); ++ts)
    {
    }
}

// Define the built in VrmlNodeType:: "Office" fields

VrmlNodeType *VrmlNodeOffice::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Office", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("applicationType", VrmlField::SFSTRING);
    t->addEventOut("command", VrmlField::SFSTRING);
    t->addEventIn("events", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeOffice::nodeType() const
{
    return defineType(0);
}

VrmlNodeOffice::VrmlNodeOffice(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_applicationType("PowerPoint")
    , d_command("")
    , d_events("")
{
    setModified();
}

void VrmlNodeOffice::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
        allOffice.push_front(this);
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeOffice::VrmlNodeOffice(const VrmlNodeOffice &n)
    : VrmlNodeChild(n.d_scene)
    , d_applicationType(n.d_applicationType)
    , d_command(n.d_command)
    , d_events(n.d_events)
{
    setModified();
}

VrmlNodeOffice::~VrmlNodeOffice()
{
    allOffice.remove(this);
}

VrmlNode *VrmlNodeOffice::cloneMe() const
{
    return new VrmlNodeOffice(*this);
}

VrmlNodeOffice *VrmlNodeOffice::toOffice() const
{
    return (VrmlNodeOffice *)this;
}

void VrmlNodeOffice::render(Viewer *viewer)
{
    (void)viewer;
}

ostream &VrmlNodeOffice::printFields(ostream &os, int indent)
{
    if (!d_applicationType.get())
        PRINT_FIELD(applicationType);
    if (!d_command.get())
        PRINT_FIELD(command);
    if (!d_events.get())
        PRINT_FIELD(events);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeOffice::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(applicationType, SFString)
    else if
        TRY_FIELD(command, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "applicationType") == 0)
    {
        for(officeList::iterator it = OfficePlugin::instance()->officeConnections.begin();it !=OfficePlugin::instance()->officeConnections.end();it++)
        {
            if((*it)->applicationType == d_applicationType.get() || (*it)->productName == d_applicationType.get())
            {
                officeConnection = (*it);
            }
        }
    }
    if (strcmp(fieldName, "command") == 0)
    {
        if(officeConnection==NULL)
        {
            for(officeList::iterator it = OfficePlugin::instance()->officeConnections.begin();it !=OfficePlugin::instance()->officeConnections.end();it++)
            {
                if((*it)->applicationType == d_applicationType.get() || (*it)->productName == d_applicationType.get())
                {
                    officeConnection = (*it);
                }
            }
        }
        if(officeConnection!=NULL)
        {
            covise::TokenBuffer stb;
            stb << fieldValue.toSFString()->get();

            Message message(stb);
            message.type = (int)OfficePlugin::MSG_String;
            officeConnection->sendMessage(message);
        }
    }
}

const VrmlField *VrmlNodeOffice::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "applicationType") == 0)
        return &d_applicationType;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
