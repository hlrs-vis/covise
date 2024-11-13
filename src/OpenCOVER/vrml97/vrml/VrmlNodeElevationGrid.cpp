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

void VrmlNodeElevationGrid::initFields(VrmlNodeElevationGrid *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                    exposedField("color", node->d_color),
                    exposedField("normal", node->d_normal),
                    exposedField("texCoord", node->d_texCoord),
                    field("ccw", node->d_ccw),
                    field("colorPerVertex", node->d_colorPerVertex),
                    field("creaseAngle", node->d_creaseAngle),
                    field("height", node->d_height),
                    field("normalPerVertex", node->d_normalPerVertex),
                    field("solid", node->d_solid),
                    field("xDimension", node->d_xDimension),
                    field("xSpacing", node->d_xSpacing),
                    field("zDimension", node->d_zDimension),
                    field("zSpacing", node->d_zSpacing), 
                    eventInCallBack("set_height", node->d_height));

}

const char *VrmlNodeElevationGrid::name() { return "ElevationGrid"; }


VrmlNodeElevationGrid::VrmlNodeElevationGrid(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
    , d_ccw(true)
    , d_colorPerVertex(true)
    , d_normalPerVertex(true)
    , d_solid(true)
    , d_xSpacing(1.0)
    , d_zSpacing(1.0)
{
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

VrmlNodeColor *VrmlNodeElevationGrid::color()
{
    return d_color.get() ? d_color.get()->as<VrmlNodeColor>() : 0;
}

Viewer::Object VrmlNodeElevationGrid::insertGeometry(Viewer *viewer)
{
    Viewer::Object obj = 0;

    if (d_height.size() > 0)
    {
        float *tc = 0, *normals = 0, *colors = 0;

        if (d_texCoord.get())
        {
            VrmlMFVec2f &texcoord = d_texCoord.get()->as<VrmlNodeTextureCoordinate>()->coordinate();
            tc = &texcoord[0][0];
        }

        if (d_normal.get())
        {
            VrmlMFVec3f &n = d_normal.get()->as<VrmlNodeNormal>()->normal();
            normals = &n[0][0];
        }

        int componentsPerColor = 3;
        VrmlNode *colorNode = d_color.get();
        if (colorNode && (strcmp(colorNode->nodeType()->getName(), "ColorRGBA") == 0))
        {
            VrmlMFColorRGBA &c = d_color.get()->as<VrmlNodeColorRGBA>()->color();
            colors = &c[0][0];
            componentsPerColor = 4;
        }
        else if (d_color.get())
        {
            VrmlMFColor &c = d_color.get()->as<VrmlNodeColor>()->color();
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
