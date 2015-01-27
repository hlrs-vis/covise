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

using namespace vrml;

// Return a new VrmlNodeSwitch
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeSwitch(s); }

// Define the built in VrmlNodeType:: "Switch" fields

VrmlNodeType *VrmlNodeSwitch::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Switch", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("choice", VrmlField::MFNODE);
    t->addExposedField("children", VrmlField::MFNODE);
    t->addExposedField("whichChoice", VrmlField::SFINT32);

    return t;
}

VrmlNodeType *VrmlNodeSwitch::nodeType() const { return defineType(0); }

VrmlNodeSwitch::VrmlNodeSwitch(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_whichChoice(-1)
{
    firstTime = true;
    forceTraversal(false);
}

VrmlNodeSwitch::~VrmlNodeSwitch()
{
}

VrmlNode *VrmlNodeSwitch::cloneMe() const
{
    return new VrmlNodeSwitch(*this);
}

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

std::ostream &VrmlNodeSwitch::printFields(std::ostream &os, int indent)
{
    if (d_choice.size() > 0)
        PRINT_FIELD(choice);
    if (d_whichChoice.get() != -1)
        PRINT_FIELD(whichChoice);
    return os;
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

// Set the value of one of the node fields.
void VrmlNodeSwitch::setField(const char *fieldName,
                              const VrmlField &fieldValue)
{
    if ((strcmp(fieldName, "children") == 0) || (strcmp(fieldName, "choice") == 0))
    {
        if (fieldValue.toMFNode())
            d_choice = (VrmlMFNode &)fieldValue;
        else
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s).\n",
                               fieldValue.fieldTypeName(), "children or choice", nodeType()->getName(), "MFNode");
    }
    else if
        TRY_FIELD(whichChoice, SFInt)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeSwitch::getField(const char *fieldName) const
{
    if ((strcmp(fieldName, "children") == 0) || (strcmp(fieldName, "choice") == 0))
        return &d_choice;
    else if (strcmp(fieldName, "whichChoice") == 0)
        return &d_whichChoice;

    return VrmlNodeChild::getField(fieldName);
}

VrmlNodeSwitch *VrmlNodeSwitch::toSwitch() const //LarryD
{
    return (VrmlNodeSwitch *)this;
}
