/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePolygonsCommon.cpp

#include <cfloat>

#include "VrmlNodePolygonsCommon.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

void VrmlNodePolygonsCommon::initFields(VrmlNodePolygonsCommon *node, VrmlNodeType *t)
{
    VrmlNodeColoredSet::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("normal", node->d_normal),
                     exposedField("texCoord", node->d_texCoord),
                     exposedField("fogCoord", node->d_fogCoord),
                     exposedField("attrib", node->d_attrib),
                     field("ccw", node->d_ccw),
                     field("normalPerVertex", node->d_normalPerVertex),
                     field("solid", node->d_solid));
}

VrmlNodePolygonsCommon::VrmlNodePolygonsCommon(VrmlScene *scene, const std::string &name)
    : VrmlNodeColoredSet(scene, name)
    , d_ccw(true)
    , d_normalPerVertex(true)
    , d_solid(true)
{
}

void VrmlNodePolygonsCommon::cloneChildren(VrmlNamespace *ns)
{
    if (d_color.get())
    {
        d_color.set(d_color.get()->clone(ns));
        d_color.get()->parentList.push_back(this);
    }
    if (d_coord.get())
    {
        d_coord.set(d_coord.get()->clone(ns));
        d_coord.get()->parentList.push_back(this);
    }
    if (d_normal.get())
    {
        d_normal.set(d_normal.get()->clone(ns));
        d_normal.get()->parentList.push_back(this);
    }
    if (d_texCoord.get())
    {
        d_texCoord.set(d_texCoord.get()->clone(ns));
        d_texCoord.get()->parentList.push_back(this);
    }
    if (d_fogCoord.get())
    {
        d_fogCoord.set(d_texCoord.get()->clone(ns));
        d_fogCoord.get()->parentList.push_back(this);
    }

    int n = d_attrib.size();
    VrmlNode **kids = d_attrib.get();
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

bool VrmlNodePolygonsCommon::isModified() const
{
    int n = d_attrib.size();
    for (int i = 0; i < n; ++i)
    {
        if (d_attrib[i])
            if (d_attrib[i]->isModified())
                return true;
    }
    return (d_modified || (d_color.get() && d_color.get()->isModified()) || (d_coord.get() && d_coord.get()->isModified()) || (d_normal.get() && d_normal.get()->isModified()) || (d_texCoord.get() && d_texCoord.get()->isModified()) || (d_fogCoord.get() && d_fogCoord.get()->isModified()));
}

void VrmlNodePolygonsCommon::clearFlags()
{
    if (d_color.get())
        d_color.get()->clearFlags();
    if (d_coord.get())
        d_coord.get()->clearFlags();
    if (d_normal.get())
        d_normal.get()->clearFlags();
    if (d_texCoord.get())
        d_texCoord.get()->clearFlags();
    if (d_fogCoord.get())
        d_fogCoord.get()->clearFlags();
    int n = d_attrib.size();
    for (int i = 0; i < n; ++i)
        if (d_attrib[i])
            d_attrib[i]->clearFlags();
    VrmlNode::clearFlags();
}

void VrmlNodePolygonsCommon::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_color.get())
        d_color.get()->addToScene(s, rel);
    if (d_coord.get())
        d_coord.get()->addToScene(s, rel);
    if (d_normal.get())
        d_normal.get()->addToScene(s, rel);
    if (d_texCoord.get())
        d_texCoord.get()->addToScene(s, rel);
    if (d_fogCoord.get())
        d_fogCoord.get()->addToScene(s, rel);

    int n = d_attrib.size();
    for (int i = 0; i < n; ++i)
        if (d_attrib[i])
            d_attrib[i]->addToScene(s, rel);

    nodeStack.pop_front();
}

void VrmlNodePolygonsCommon::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_color.get())
        d_color.get()->copyRoutes(ns);
    if (d_coord.get())
        d_coord.get()->copyRoutes(ns);
    if (d_normal.get())
        d_normal.get()->copyRoutes(ns);
    if (d_texCoord.get())
        d_texCoord.get()->copyRoutes(ns);
    if (d_fogCoord.get())
        d_fogCoord.get()->copyRoutes(ns);
    // Copy attribs' routes
    int n = d_attrib.size();
    for (int i = 0; i < n; ++i)
        if (d_attrib[i])
            d_attrib[i]->copyRoutes(ns);
    nodeStack.pop_front();
}

VrmlNode *VrmlNodePolygonsCommon::getNormal()
{
    return d_normal.get();
}

VrmlNode *VrmlNodePolygonsCommon::getTexCoord()
{
    return d_texCoord.get();
}
