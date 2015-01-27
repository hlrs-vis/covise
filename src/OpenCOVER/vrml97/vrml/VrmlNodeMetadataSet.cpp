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

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeMetadataSet(s); }

// Define the built in VrmlNodeType:: "MetadataSet" fields

VrmlNodeType *VrmlNodeMetadataSet::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("MetadataSet", creator);
    }

    VrmlNodeMetadata::defineType(t); // Parent class

    t->addExposedField("value", VrmlField::MFNODE);

    return t;
}

VrmlNodeType *VrmlNodeMetadataSet::nodeType() const { return defineType(0); }

VrmlNodeMetadataSet::VrmlNodeMetadataSet(VrmlScene *scene)
    : VrmlNodeMetadata(scene)
{
}

VrmlNodeMetadataSet::~VrmlNodeMetadataSet()
{
}

VrmlNode *VrmlNodeMetadataSet::cloneMe() const
{
    return new VrmlNodeMetadataSet(*this);
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

std::ostream &VrmlNodeMetadataSet::printFields(std::ostream &os, int indent)
{
    if (!d_value.get())
        PRINT_FIELD(value);

    VrmlNodeMetadata::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMetadataSet::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    if (strcmp(fieldName, "value") == 0)
    {
        if (fieldValue.toMFNode())
            d_value = (VrmlMFNode &)fieldValue;
        else
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s).\n",
                               fieldValue.fieldTypeName(), "value", nodeType()->getName(), "MFNode");
    }
    else
        VrmlNodeMetadata::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeMetadataSet::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "value") == 0)
        return &d_value;
    return VrmlNodeMetadata::getField(fieldName);
}

VrmlNodeMetadataSet *VrmlNodeMetadataSet::toMetadataSet() const
{
    return (VrmlNodeMetadataSet *)this;
}
