/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeFog.cpp

#include "VrmlNodeFog.h"

#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "VrmlSFBool.h"
#include "Viewer.h"

using std::cerr;
using std::endl;
using namespace vrml;

//  Fog factory.
//  Since Fog is a bindable child node, the first one created needs
//  to notify its containing scene.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeFog(scene);
}

// Define the built in VrmlNodeType:: "Fog" fields
void VrmlNodeFog::initFields(VrmlNodeFog *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     field("color", node->d_color),
                     field("fogType", node->d_fogType),
                     field("visibilityRange", node->d_visibilityRange));
    if(t)
    {
        t->addEventIn("set_bind", VrmlField::SFBOOL);
        t->addEventOut("isBound", VrmlField::SFBOOL);
    }
    VrmlNodeChild::initFields(node, t);
}

const char *VrmlNodeFog::name() { return "Fog"; }

VrmlNodeFog::VrmlNodeFog(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_color(1.0, 1.0, 1.0)
    , d_fogType("LINEAR")
    , d_visibilityRange(0.0)
{
    if (d_scene)
        d_scene->addFog(this);
}

VrmlNodeFog::~VrmlNodeFog()
{
    if (d_scene)
        d_scene->removeFog(this);
}

void VrmlNodeFog::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addFog(this);
}

void VrmlNodeFog::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
    if (strcmp(eventName, "set_bind") == 0)
    {
        VrmlNodeFog *current = d_scene->bindableFogTop();
        const VrmlSFBool *b = fieldValue->toSFBool();

        if (!b)
        {
            cerr << "Error: invalid value for Fog::set_bind eventIn "
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
                current = d_scene->bindableFogTop();
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
