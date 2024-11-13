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
#include <vrml97/vrml/System.h>
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

void VrmlNodeOffice::update()
{
    list<VrmlNodeOffice *>::iterator ts;
    for (ts = allOffice.begin(); ts != allOffice.end(); ++ts)
    {
    }
}

// Define the built in VrmlNodeType:: "Office" fields

void VrmlNodeOffice::initFields(VrmlNodeOffice *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("applicationType", node->d_applicationType, [node](auto f){
                        for(officeList::iterator it = OfficePlugin::instance()->officeConnections.begin();it !=OfficePlugin::instance()->officeConnections.end();it++)
                        {
                            if((*it)->applicationType == node->d_applicationType.get() || (*it)->productName == node->d_applicationType.get())
                            {
                                node->officeConnection = (*it);
                            }
                        }                           
                     }), 
                     exposedField("command", node->d_command, [node](auto f){
                        if(node->officeConnection==NULL)
                        {
                            for(officeList::iterator it = OfficePlugin::instance()->officeConnections.begin();it !=OfficePlugin::instance()->officeConnections.end();it++)
                            {
                                if((*it)->applicationType == node->d_applicationType.get() || (*it)->productName == node->d_applicationType.get())
                                {
                                    node->officeConnection = (*it);
                                }
                            }
                        }
                        if(node->officeConnection!=NULL)
                        {
                            covise::TokenBuffer stb;
                            stb << node->d_command.get();

                            Message message(stb);
                            message.type = (int)OfficePlugin::MSG_String;
                            node->officeConnection->sendMessage(message);
                        }
                     }),
                     eventInCallBack("events", node->d_events));
}

const char *VrmlNodeOffice::name()
{
    return "Office";
}

VrmlNodeOffice::VrmlNodeOffice(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
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
    : VrmlNodeChild(n)
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

void VrmlNodeOffice::render(Viewer *viewer)
{
    (void)viewer;
}
