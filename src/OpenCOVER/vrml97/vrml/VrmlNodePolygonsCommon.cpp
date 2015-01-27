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

// Define the built in VrmlNodeType:: "IndexedPolygons*Set" fields

VrmlNodeType *VrmlNodePolygonsCommon::defineType(VrmlNodeType *t)
{
    if (!t)
    {
        return NULL;
    }
    VrmlNodeColoredSet::defineType(t); // Parent class

    t->addExposedField("normal", VrmlField::SFNODE);
    t->addExposedField("texCoord", VrmlField::SFNODE);
    t->addExposedField("fogCoord", VrmlField::SFNODE);
    t->addExposedField("attrib", VrmlField::MFNODE);
    t->addField("ccw", VrmlField::SFBOOL);
    t->addField("normalPerVertex", VrmlField::SFBOOL);
    t->addField("solid", VrmlField::SFBOOL);

    return t;
}

VrmlNodePolygonsCommon::VrmlNodePolygonsCommon(VrmlScene *scene)
    : VrmlNodeColoredSet(scene)
    , d_ccw(true)
    , d_normalPerVertex(true)
    , d_solid(true)
{
}

VrmlNodePolygonsCommon::~VrmlNodePolygonsCommon()
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

std::ostream &VrmlNodePolygonsCommon::printFields(std::ostream &os, int indent)
{
    if (!d_ccw.get())
        PRINT_FIELD(ccw);
    if (!d_normalPerVertex.get())
        PRINT_FIELD(normalPerVertex);
    if (!d_solid.get())
        PRINT_FIELD(solid);

    if (d_normal.get())
        PRINT_FIELD(normal);
    if (d_color.get())
        PRINT_FIELD(color);
    if (d_texCoord.get())
        PRINT_FIELD(texCoord);

    if (d_fogCoord.get())
        PRINT_FIELD(fogCoord);
    if (d_attrib.size() > 0)
        PRINT_FIELD(attrib);

    VrmlNodeColoredSet::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodePolygonsCommon::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    if
        TRY_FIELD(ccw, SFBool)
    else if
        TRY_SFNODE_FIELD(normal, Normal)
    else if
        TRY_FIELD(normalPerVertex, SFBool)
    else if
        TRY_FIELD(solid, SFBool)
    else if
        TRY_SFNODE_FIELD3(texCoord, TextureCoordinate, MultiTextureCoordinate, TextureCoordinateGenerator)
    else
        VrmlNodeColoredSet::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodePolygonsCommon::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "ccw") == 0)
        return &d_ccw;
    else if (strcmp(fieldName, "normal") == 0)
        return &d_normal;
    else if (strcmp(fieldName, "normalPerVertex") == 0)
        return &d_normalPerVertex;
    else if (strcmp(fieldName, "solid") == 0)
        return &d_solid;
    else if (strcmp(fieldName, "texCoord") == 0)
        return &d_texCoord;
    else if (strcmp(fieldName, "colorPerVertex") == 0)
        return &d_colorPerVertex;

    return VrmlNodeColoredSet::getField(fieldName);
}

VrmlNode *VrmlNodePolygonsCommon::getNormal()
{
    return d_normal.get();
}

VrmlNode *VrmlNodePolygonsCommon::getTexCoord()
{
    return d_texCoord.get();
}
