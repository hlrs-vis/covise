/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIndexedSet.cpp

#include "VrmlNodeIndexedSet.h"

#include "VrmlNodeType.h"

using namespace vrml;

void VrmlNodeIndexedSet::initFields(VrmlNodeIndexedSet *node, VrmlNodeType *t)
{
    VrmlNodeColoredSet::initFields(node, t); // Parent class

    initFieldsHelper(node, t,
                     field("colorIndex", node->d_colorIndex),
                     field("coordIndex", node->d_coordIndex));

    if (t)
    {
        t->addEventIn("set_colorIndex", VrmlField::MFINT32);
        t->addEventIn("set_coordIndex", VrmlField::MFINT32);
    }
}

VrmlNodeIndexedSet::VrmlNodeIndexedSet(VrmlScene *scene, const std::string &name)
    : VrmlNodeColoredSet(scene, name)
{
}

bool VrmlNodeIndexedSet::isModified() const
{
    return (d_modified);
}

void VrmlNodeIndexedSet::clearFlags()
{
    VrmlNode::clearFlags();
}

void VrmlNodeIndexedSet::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    nodeStack.pop_front();
}

void VrmlNodeIndexedSet::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    nodeStack.pop_front();
}

const VrmlMFInt &VrmlNodeIndexedSet::getCoordIndex() const
{
    return d_coordIndex;
}

// LarryD Feb 18/99
const VrmlMFInt &VrmlNodeIndexedSet::getColorIndex() const
{
    return d_colorIndex;
}
