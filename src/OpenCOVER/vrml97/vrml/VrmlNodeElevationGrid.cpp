/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeElevationGrid.cpp

#include "VrmlNodeElevationGrid.h"

#include "VrmlNodeType.h"

#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeNormal.h"
#include "VrmlNodeTextureCoordinate.h"

#include "Viewer.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeElevationGrid(s);
}

// Define the built in VrmlNodeType:: "ElevationGrid" fields

VrmlNodeType *VrmlNodeElevationGrid::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("ElevationGrid", creator);
    }

    VrmlNodeGeometry::defineType(t); // Parent class

    t->addEventIn("set_height", VrmlField::MFFLOAT);
    t->addExposedField("color", VrmlField::SFNODE);
    t->addExposedField("normal", VrmlField::SFNODE);
    t->addExposedField("texCoord", VrmlField::SFNODE);
    t->addField("ccw", VrmlField::SFBOOL);
    t->addField("colorPerVertex", VrmlField::SFBOOL);
    t->addField("creaseAngle", VrmlField::SFFLOAT);
    t->addField("height", VrmlField::MFFLOAT);
    t->addField("normalPerVertex", VrmlField::SFBOOL);
    t->addField("solid", VrmlField::SFBOOL);
    t->addField("xDimension", VrmlField::SFINT32);
    t->addField("xSpacing", VrmlField::SFFLOAT);
    t->addField("zDimension", VrmlField::SFINT32);
    t->addField("zSpacing", VrmlField::SFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeElevationGrid::nodeType() const { return defineType(0); }

VrmlNodeElevationGrid::VrmlNodeElevationGrid(VrmlScene *scene)
    : VrmlNodeGeometry(scene)
    , d_ccw(true)
    , d_colorPerVertex(true)
    , d_normalPerVertex(true)
    , d_solid(true)
    , d_xSpacing(1.0)
    , d_zSpacing(1.0)
{
}

VrmlNodeElevationGrid::~VrmlNodeElevationGrid()
{
}

VrmlNode *VrmlNodeElevationGrid::cloneMe() const
{
    return new VrmlNodeElevationGrid(*this);
}

void VrmlNodeElevationGrid::cloneChildren(VrmlNamespace *ns)
{
    if (d_color.get())
    {
        d_color.set(d_color.get()->clone(ns));
        d_color.get()->parentList.push_back(this);
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
}

bool VrmlNodeElevationGrid::isModified() const
{
    return (d_modified || (d_color.get() && d_color.get()->isModified()) || (d_normal.get() && d_normal.get()->isModified()) || (d_texCoord.get() && d_texCoord.get()->isModified()));
}

void VrmlNodeElevationGrid::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_color.get())
        d_color.get()->clearFlags();
    if (d_normal.get())
        d_normal.get()->clearFlags();
    if (d_texCoord.get())
        d_texCoord.get()->clearFlags();
}

void VrmlNodeElevationGrid::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_color.get())
        d_color.get()->addToScene(s, rel);
    if (d_normal.get())
        d_normal.get()->addToScene(s, rel);
    if (d_texCoord.get())
        d_texCoord.get()->addToScene(s, rel);
    nodeStack.pop_front();
}

void VrmlNodeElevationGrid::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_color.get())
        d_color.get()->copyRoutes(ns);
    if (d_normal.get())
        d_normal.get()->copyRoutes(ns);
    if (d_texCoord.get())
        d_texCoord.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

std::ostream &VrmlNodeElevationGrid::printFields(std::ostream &os, int indent)
{
    if (d_color.get())
        PRINT_FIELD(color);
    if (d_normal.get())
        PRINT_FIELD(normal);
    if (d_texCoord.get())
        PRINT_FIELD(texCoord);

    if (!d_ccw.get())
        PRINT_FIELD(ccw);
    if (!d_colorPerVertex.get())
        PRINT_FIELD(colorPerVertex);
    if (!d_normalPerVertex.get())
        PRINT_FIELD(normalPerVertex);
    if (!d_solid.get())
        PRINT_FIELD(solid);

    if (d_creaseAngle.get() != 0.0)
        PRINT_FIELD(creaseAngle);
    if (d_height.size() > 0)
        PRINT_FIELD(height);

    if (d_xDimension.get() != 0)
        PRINT_FIELD(xDimension);
    if (d_xSpacing.get() != 0)
        PRINT_FIELD(xSpacing);
    if (d_zDimension.get() != 0)
        PRINT_FIELD(zDimension);
    if (d_zSpacing.get() != 0)
        PRINT_FIELD(zSpacing);

    return os;
}

VrmlNodeColor *VrmlNodeElevationGrid::color()
{
    return d_color.get() ? d_color.get()->toColor() : 0;
}

Viewer::Object VrmlNodeElevationGrid::insertGeometry(Viewer *viewer)
{
    Viewer::Object obj = 0;

    if (d_height.size() > 0)
    {
        float *tc = 0, *normals = 0, *colors = 0;

        if (d_texCoord.get())
        {
            VrmlMFVec2f &texcoord = d_texCoord.get()->toTextureCoordinate()->coordinate();
            tc = &texcoord[0][0];
        }

        if (d_normal.get())
        {
            VrmlMFVec3f &n = d_normal.get()->toNormal()->normal();
            normals = &n[0][0];
        }

        int componentsPerColor = 3;
        VrmlNode *colorNode = d_color.get();
        if (colorNode && (strcmp(colorNode->nodeType()->getName(), "ColorRGBA") == 0))
        {
            VrmlMFColorRGBA &c = d_color.get()->toColorRGBA()->color();
            colors = &c[0][0];
            componentsPerColor = 4;
        }
        else if (d_color.get())
        {
            VrmlMFColor &c = d_color.get()->toColor()->color();
            colors = &c[0][0];
        }

        // insert geometry
        unsigned int optMask = 0;
        if (d_ccw.get())
            optMask |= Viewer::MASK_CCW;
        if (d_solid.get())
            optMask |= Viewer::MASK_SOLID;
        if (d_colorPerVertex.get())
            optMask |= Viewer::MASK_COLOR_PER_VERTEX;
        if (d_normalPerVertex.get())
            optMask |= Viewer::MASK_NORMAL_PER_VERTEX;
        if (componentsPerColor == 4)
            optMask |= Viewer::MASK_COLOR_RGBA;

        obj = viewer->insertElevationGrid(optMask,
                                          d_xDimension.get(),
                                          d_zDimension.get(),
                                          d_height.get(),
                                          d_xSpacing.get(),
                                          d_zSpacing.get(),
                                          tc,
                                          normals,
                                          colors,
                                          d_creaseAngle.get());
    }

    if (d_color.get())
        d_color.get()->clearModified();
    if (d_normal.get())
        d_normal.get()->clearModified();
    if (d_texCoord.get())
        d_texCoord.get()->clearModified();

    return obj;
}

// Set the value of one of the node fields.

void VrmlNodeElevationGrid::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_SFNODE_FIELD2(color, Color, ColorRGBA)
    else if
        TRY_SFNODE_FIELD(normal, Normal)
    else if
        TRY_SFNODE_FIELD(texCoord, TextureCoordinate)
    else if
        TRY_FIELD(ccw, SFBool)
    else if
        TRY_FIELD(colorPerVertex, SFBool)
    else if
        TRY_FIELD(creaseAngle, SFFloat)
    else if
        TRY_FIELD(height, MFFloat)
    else if
        TRY_FIELD(normalPerVertex, SFBool)
    else if
        TRY_FIELD(solid, SFBool)
    else if
        TRY_FIELD(xDimension, SFInt)
    else if
        TRY_FIELD(xSpacing, SFFloat)
    else if
        TRY_FIELD(zDimension, SFInt)
    else if
        TRY_FIELD(zSpacing, SFFloat)
    else
        VrmlNodeGeometry::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeElevationGrid::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "color") == 0)
        return &d_color;
    else if (strcmp(fieldName, "normal") == 0)
        return &d_normal;
    else if (strcmp(fieldName, "texCoord") == 0)
        return &d_texCoord;
    else if (strcmp(fieldName, "ccw") == 0)
        return &d_ccw;
    else if (strcmp(fieldName, "colorPerVertex") == 0)
        return &d_colorPerVertex;
    else if (strcmp(fieldName, "creaseAngle") == 0)
        return &d_creaseAngle;
    else if (strcmp(fieldName, "height") == 0)
        return &d_height;
    else if (strcmp(fieldName, "normalPerVertex") == 0)
        return &d_normalPerVertex;
    else if (strcmp(fieldName, "solid") == 0)
        return &d_solid;
    else if (strcmp(fieldName, "xDimension") == 0)
        return &d_xDimension;
    else if (strcmp(fieldName, "xSpacing") == 0)
        return &d_xSpacing;
    else if (strcmp(fieldName, "zDimension") == 0)
        return &d_zDimension;
    else if (strcmp(fieldName, "zSpacing") == 0)
        return &d_zSpacing;

    return VrmlNodeGeometry::getField(fieldName);
}

// LarryD Mar 09/99
VrmlNodeElevationGrid *VrmlNodeElevationGrid::toElevationGrid() const
{
    return (VrmlNodeElevationGrid *)this;
}

VrmlNode *VrmlNodeElevationGrid::getNormal() // LarryD Mar 09/99
{
    return d_normal.get();
}

VrmlNode *VrmlNodeElevationGrid::getTexCoord() // LarryD Mar 09/99
{
    return d_texCoord.get();
}

// LarryD Mar 09/99
const VrmlMFFloat &VrmlNodeElevationGrid::getHeight() const
{
    return d_height;
}
