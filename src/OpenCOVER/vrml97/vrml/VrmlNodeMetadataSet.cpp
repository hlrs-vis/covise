/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataSet.cpp

#include "VrmlNodeMetadataSet.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

void VrmlNodeMetadataSet::initFields(VrmlNodeMetadataSet *node, VrmlNodeType *t)
{
    VrmlNodeMetadata::initFields(node, t);
    initFieldsHelper(node, t, exposedField("value", node->d_value));
}

const char *VrmlNodeMetadataSet::name() { return "MetadataSet"; }

VrmlNodeMetadataSet::VrmlNodeMetadataSet(VrmlScene *scene)
    : VrmlNodeMetadata(scene, name())
{
}

void VrmlNodeMetadataSet::cloneChildren(VrmlNamespace *ns)
{
    int n = d_value.size();
    VrmlNode **kids = d_value.get();
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

bool VrmlNodeMetadataSet::isModified() const
{
    if (d_modified)
        return true;

    int n = d_value.size();

    for (int i = 0; i < n; ++i)
    {
        if (d_value[i] == NULL)
            return false;
        if (d_value[i]->isModified())
            return true;
    }

    return false;
}

void VrmlNodeMetadataSet::clearFlags()
{
    VrmlNode::clearFlags();

    int n = d_value.size();
    for (int i = 0; i < n; ++i)
        if (d_value[i])
            d_value[i]->clearFlags();
}

void VrmlNodeMetadataSet::addToScene(VrmlScene *s, const char *relativeUrl)
{
    d_scene = s;

    nodeStack.push_front(this);
    System::the->debug("VrmlNodeMetadataSet::addToScene( %s )\n",
                       relativeUrl ? relativeUrl : "<null>");

    int n = d_value.size();

    for (int i = 0; i < n; ++i)
    {
        if (d_value[i])
            d_value[i]->addToScene(s, relativeUrl);
    }
    nodeStack.pop_front();
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeMetadataSet::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns); // Copy my routes

    // Copy values' routes
    int n = d_value.size();
    for (int i = 0; i < n; ++i)
        if (d_value[i])
            d_value[i]->copyRoutes(ns);
    nodeStack.pop_front();
}

VrmlNodeMetadataSet *VrmlNodeMetadataSet::toMetadataSet() const
{
    return (VrmlNodeMetadataSet *)this;
}
