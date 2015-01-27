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
#include "Viewer.h"

using namespace vrml;

// Return a new VrmlNodeLOD
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeLOD(s); }

// Define the built in VrmlNodeType:: "LOD" fields

VrmlNodeType *VrmlNodeLOD::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("LOD", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("level", VrmlField::MFNODE);
    t->addExposedField("children", VrmlField::MFNODE);
    t->addField("center", VrmlField::SFVEC3F);
    t->addField("range", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeLOD::nodeType() const { return defineType(0); }

VrmlNodeLOD::VrmlNodeLOD(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    firstTime = true;
    forceTraversal(false);
}

VrmlNodeLOD::~VrmlNodeLOD()
{
}

VrmlNode *VrmlNodeLOD::cloneMe() const
{
    return new VrmlNodeLOD(*this);
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

std::ostream &VrmlNodeLOD::printFields(std::ostream &os, int indent)
{
    if (d_level.size() > 0)
        PRINT_FIELD(level);
    if (!FPZERO(d_center.x()) || !FPZERO(d_center.y()) || !FPZERO(d_center.z()))
        PRINT_FIELD(center);

    if (d_range.size() > 0)
        PRINT_FIELD(range);

    return os;
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

// Set the value of one of the node fields.
void VrmlNodeLOD::setField(const char *fieldName, const VrmlField &fieldValue)
{
    // check against both fieldnames cause scene() and getNamespace() maybe NULL
    // no easy way to check X3D status
    if ((strcmp(fieldName, "children") == 0) || (strcmp(fieldName, "level") == 0))
    {
        if (fieldValue.toMFNode())
            d_level = (VrmlMFNode &)fieldValue;
        else
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s).\n",
                               fieldValue.fieldTypeName(), "children or level", nodeType()->getName(), "MFNode");
    }
    else if
        TRY_FIELD(center, SFVec3f)
    else if
        TRY_FIELD(range, MFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeLOD::getField(const char *fieldName) const
{
    // check against both fieldnames cause scene() and getNamespace() maybe NULL
    // no easy way to check X3D status
    if ((strcmp(fieldName, "children") == 0) || (strcmp(fieldName, "level") == 0))
        return &d_level;
    else if (strcmp(fieldName, "center") == 0)
        return &d_center;
    else if (strcmp(fieldName, "range") == 0)
        return &d_range;

    return VrmlNode::getField(fieldName);
}

VrmlNodeLOD *VrmlNodeLOD::toLOD() const
{
    return (VrmlNodeLOD *)this;
}

const VrmlMFFloat &VrmlNodeLOD::getRange() const
{
    return d_range;
}

const VrmlSFVec3f &VrmlNodeLOD::getCenter() const
{
    return d_center;
}
