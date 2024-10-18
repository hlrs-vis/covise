/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNavigationInfo.cpp

#include "VrmlNodeNavigationInfo.h"

#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"

using std::cerr;
using std::endl;
using namespace vrml;

//  NavigationInfo factory.
//  Since NavInfo is a bindable child node, the first one created needs
//  to notify its containing scene.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeNavigationInfo(scene);
}

// Define the built in VrmlNodeType:: "NavigationInfo" fields

void VrmlNodeNavigationInfo::initFields(VrmlNodeNavigationInfo *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); 
    initFieldsHelper(node, t,
                     exposedField("avatarSize", node->d_avatarSize),
                     exposedField("headlight", node->d_headlight),
                     exposedField("speed", node->d_speed),
                     exposedField("scale", node->d_scale),
                     exposedField("near", node->d_near),
                     exposedField("far", node->d_far),
                     exposedField("type", node->d_type),
                     exposedField("transitionType", node->d_transitionType),
                     exposedField("transitionTime", node->d_transitionTime),
                     exposedField("visibilityLimit", node->d_visibilityLimit));
    if(t)
    {
        t->addEventIn("set_bind", VrmlField::SFBOOL);
        t->addEventIn("set_bindLast", VrmlField::SFBOOL);
        t->addEventOut("isBound", VrmlField::SFBOOL);
    }                    
}

const char *VrmlNodeNavigationInfo::name() { return "NavigationInfo"; }

VrmlNodeNavigationInfo::VrmlNodeNavigationInfo(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , lastBind(true)
    , d_headlight(true)
    , d_scale(-1.0)
    , d_speed(1.0)
    , d_near(-1.0)
    , d_far(-1.0)
    , d_visibilityLimit(0.0)
{
    float avatarSize[] = { 0.25f, 1.6f, 0.75f };
    float transitionTime[] = { 1.0f };
    const char *type[] = { "WALK", "ANY" };
    const char *transitionType[] = { "ANIMATE" };

    d_transitionType.set(1, transitionType);
    d_transitionTime.set(1, transitionTime);
    d_avatarSize.set(3, avatarSize);
    d_type.set(2, type);
    if (d_scene)
        d_scene->addNavigationInfo(this);
}

VrmlNodeNavigationInfo::~VrmlNodeNavigationInfo()
{
    if (d_scene)
        d_scene->removeNavigationInfo(this);
}

VrmlNodeNavigationInfo *VrmlNodeNavigationInfo::toNavigationInfo() const
{
    return (VrmlNodeNavigationInfo *)this;
}

void VrmlNodeNavigationInfo::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addNavigationInfo(this);
}

void VrmlNodeNavigationInfo::eventIn(double timeStamp,
                                     const char *eventName,
                                     const VrmlField *fieldValue)
{
    if ((strcmp(eventName, "set_bind") == 0) || (strcmp(eventName, "set_bindLast") == 0))
    {
        if (strcmp(eventName, "set_bindLast") == 0)
            lastBind = true;
        else
            lastBind = false;
        VrmlNodeNavigationInfo *current = d_scene->bindableNavigationInfoTop();
        const VrmlSFBool *b = fieldValue->toSFBool();

        if (!b)
        {
            cerr << "Error: invalid value for NavigationInfo::set_bind eventIn "
                 << (*fieldValue) << endl;
            return;
        }

        if (b->get()) // set_bind TRUE
        {
            if (this != current)
            {
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(false));
                d_scene->bindablePush(this);
                eventOut(timeStamp, "isBound", VrmlSFBool(true));
            }
        }
        else // set_bind FALSE
        {
            d_scene->bindableRemove(this);
            if (this == current)
            {
                eventOut(timeStamp, "isBound", VrmlSFBool(false));
                current = d_scene->bindableNavigationInfoTop();
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(true));
            }
        }
    }

    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}
