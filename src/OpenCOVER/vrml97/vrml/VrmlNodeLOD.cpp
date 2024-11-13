/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeLOD.cpp

#include "VrmlNodeLOD.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"

#include "MathUtils.h"
#include "System.h"
#include "Viewer.h"

using namespace vrml;

// Return a new VrmlNodeLOD
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeLOD(s); }

// Define the built in VrmlNodeType:: "LOD" fields

void VrmlNodeLOD::initFields(VrmlNodeLOD *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("level", node->d_level),
                     exposedField("children", node->d_level),
                     field("center", node->d_center),
                     field("range", node->d_range));
}

const char *VrmlNodeLOD::name() { return "LOD"; }


VrmlNodeLOD::VrmlNodeLOD(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
    firstTime = true;
    forceTraversal(false);
}

void VrmlNodeLOD::cloneChildren(VrmlNamespace *ns)
{
    int n = d_level.size();
    VrmlNode **kids = d_level.get();
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

bool VrmlNodeLOD::isModified() const
{
    return true;
    if (d_modified)
        return true;

    int n = d_level.size();

    // This should really check which range is being rendered...
    for (int i = 0; i < n; ++i)
        if (d_level[i]->isModified())
            return true;

    return false;
}

void VrmlNodeLOD::clearFlags()
{
    VrmlNode::clearFlags();
    int n = d_level.size();
    for (int i = 0; i < n; ++i)
        d_level[i]->clearFlags();
}

void VrmlNodeLOD::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;

    int n = d_level.size();

    for (int i = 0; i < n; ++i)
        d_level[i]->addToScene(s, rel);
    nodeStack.pop_front();
}

void VrmlNodeLOD::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);

    int n = d_level.size();
    for (int i = 0; i < n; ++i)
        d_level[i]->copyRoutes(ns);
    nodeStack.pop_front();
}

// Render one of the children

void VrmlNodeLOD::render(Viewer *viewer)
{
    //clearModified();
    if (d_level.size() <= 0)
        return;

    float x, y, z;
    viewer->getPosition(&x, &y, &z);

    float dx = x - d_center.x();
    float dy = y - d_center.y();
    float dz = z - d_center.z();
    float d2 = dx * dx + dy * dy + dz * dz;
    d2 *= System::the->getLODScale();
    d2 *= System::the->getLODScale();
    int i, n = d_range.size();
    for (i = 0; i < n; ++i)
        if (d2 < d_range[i] * d_range[i])
            break;
    //fprintf(stderr,"%s d: %12.4f dx: %8.4f dy: %8.4f dz: %8.4f -> level: %d\n",name(),d2,dx,dy,dz,i);

    // Should choose an "optimal" level...
    //if (d_range.size() == 0) i = d_level.size() - 1;
    if (d_range.size() == 0)
        i = 0;

    // Not enough levels...
    if (i >= d_level.size())
        i = d_level.size() - 1;

    //printf("LOD d %g level %d\n", sqrt(d2), i);
    if (firstTime)
    {
        firstTime = false;
        int k;
        for (k = 0; k < n; ++k)
        {
            viewer->beginObject(name(), 0, this);
            viewer->setChoice(k);
            d_level[k]->render(viewer);
            viewer->endObject();
        }
        viewer->beginObject(name(), 0, this);
        viewer->setChoice(i);
    }
    else
    {
        viewer->beginObject(name(), 0, this);
        viewer->setChoice(i);
        d_level[i]->render(viewer);
    }
    // Don't re-render on their accounts
    n = d_level.size();
    for (i = 0; i < n; ++i)
        d_level[i]->clearModified();
    viewer->endObject();
}

const VrmlMFFloat &VrmlNodeLOD::getRange() const
{
    return d_range;
}

const VrmlSFVec3f &VrmlNodeLOD::getCenter() const
{
    return d_center;
}
