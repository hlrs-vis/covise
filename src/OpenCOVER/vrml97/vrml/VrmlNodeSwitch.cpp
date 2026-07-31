/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSwitch.cpp

#include "VrmlNodeSwitch.h"
#include "VrmlNodeType.h"
#include "Viewer.h"
#include "System.h"

#include <vrb/client/SharedState.h>


using namespace vrml;

void VrmlNodeSwitch::initFields(VrmlNodeSwitch *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("choice", &VrmlNodeSwitch::d_choice),
                     exposedField("children", &VrmlNodeSwitch::d_choice),
                     exposedField("shared", &VrmlNodeSwitch::d_shared),
                     exposedField("whichChoice", &VrmlNodeSwitch::d_whichChoice, [node](auto f) {
                         if (node->sharedState)
                         {
                             *node->sharedState = node->d_whichChoice.get();
                         }
                     }));
}

const char *VrmlNodeSwitch::typeName() { return "Switch"; }


VrmlNodeSwitch::VrmlNodeSwitch(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
    , d_whichChoice(-1)
{
    firstTime = true;
    forceTraversal(false);
}

VrmlNodeSwitch::VrmlNodeSwitch(const VrmlNodeSwitch &other)
    : VrmlNodeChild(other)
    , d_whichChoice(other.d_whichChoice)
{
    firstTime = true;
    forceTraversal(false);
}

VrmlNodeSwitch::~VrmlNodeSwitch(){}

void VrmlNodeSwitch::cloneChildren(VrmlNamespace *ns)
{
    int n = d_choice.size();
    VrmlNode **kids = d_choice.get();
    for (int i = 0; i < n; ++i)
    {
        if (!kids[i])
            continue;
        VrmlNode *newKid = kids[i]->clone(ns)->reference();
        kids[i]->dereference();
        kids[i] = newKid;
        kids[i]->parentList.push_back(this);
    }
}

bool VrmlNodeSwitch::isModified() const
{
    if (d_modified)
        return true;

    int w = d_whichChoice.get();

    return (w >= 0 && w < d_choice.size() && d_choice[w]->isModified());
}

void VrmlNodeSwitch::clearFlags()
{
    VrmlNode::clearFlags();

    int n = d_choice.size();
    for (int i = 0; i < n; ++i)
        d_choice[i]->clearFlags();
}

void VrmlNodeSwitch::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;

    int n = d_choice.size();

    for (int i = 0; i < n; ++i)
        d_choice[i]->addToScene(s, rel);
    nodeStack.pop_front();
}

void VrmlNodeSwitch::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);

    int n = d_choice.size();
    for (int i = 0; i < n; ++i)
        d_choice[i]->copyRoutes(ns);
    nodeStack.pop_front();
}

// Render the selected child

void VrmlNodeSwitch::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    int w = d_whichChoice.get();

    if (firstTime && (System::the->getPreloadSwitch() || (w != -1)))
    {
        firstTime = false;
        
        std::string sName(std::string(typeName()) + "." + name());

        sharedState.reset(new vrb::SharedState<int>(sName, d_whichChoice.get()));
        sharedState->setUpdateFunction([this]() {
            if (d_shared.get())
            {
                d_whichChoice = sharedState->value();
                eventOut(System::the->time(), "whichChoice", d_whichChoice);
                d_modified = true;
            }
            });
        
        
        int n;
        for (n = 0; n < d_choice.size(); ++n)
        {
            viewer->beginObject(name(), 0, this);
            viewer->setChoice(n);
            d_choice[n]->render(viewer);
            viewer->endObject();
        }
        viewer->beginObject(name(), 0, this);
        if (w < d_choice.size())
        {
            viewer->setChoice(w);
        }
        viewer->endObject();
    }
    else
    {
        viewer->beginObject(name(), 0, this);

        if (w < d_choice.size())
        {
            viewer->setChoice(w);
            if (w >= 0)
            {
                d_choice[w]->render(viewer);
            }
        }
        viewer->endObject();
    }

    clearModified();
}

// Cache a pointer to (one of the) parent transforms for proper
// rendering of bindables.

void VrmlNodeSwitch::accumulateTransform(VrmlNode *parent)
{

    int i, n = d_choice.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_choice[i];
        kid->accumulateTransform(parent);
    }
}
