/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePointSet.cpp

#include "VrmlNodePointSet.h"

#include "VrmlNodeCoordinate.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodePointSet::initFields(VrmlNodePointSet *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); 
    initFieldsHelper(node, t,
                        exposedField("color", node->d_color),
                        exposedField("coord", node->d_coord));

}

const char *VrmlNodePointSet::name() { return "PointSet"; }


VrmlNodePointSet::VrmlNodePointSet(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
{
}

void VrmlNodePointSet::cloneChildren(VrmlNamespace *ns)
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
}

bool VrmlNodePointSet::isModified() const
{
    return (d_modified || (d_color.get() && d_color.get()->isModified()) || (d_coord.get() && d_coord.get()->isModified()));
}

void VrmlNodePointSet::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_color.get())
        d_color.get()->clearFlags();
    if (d_coord.get())
        d_coord.get()->clearFlags();
}

void VrmlNodePointSet::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_color.get())
        d_color.get()->addToScene(s, rel);
    if (d_coord.get())
        d_coord.get()->addToScene(s, rel);
    nodeStack.pop_front();
}

void VrmlNodePointSet::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_color.get())
        d_color.get()->copyRoutes(ns);
    if (d_coord.get())
        d_coord.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

Viewer::Object VrmlNodePointSet::insertGeometry(Viewer *viewer)
{
    Viewer::Object obj = 0;

    if (d_coord.get())
    {
        float *color = NULL;
        int componentsPerColor = 3;
        VrmlNode *colorNode = d_color.get();
        if (colorNode && (strcmp(colorNode->nodeType()->getName(), "ColorRGBA") == 0))
        {
            VrmlMFColorRGBA &c = d_color.get()->toColorRGBA()->color();
            color = &c[0][0];
            componentsPerColor = 4;
        }
        else if (d_color.get())
        {
            VrmlMFColor &c = d_color.get()->toColor()->color();
            color = &c[0][0];
        }

        VrmlMFVec3f &coord = d_coord.get()->toCoordinate()->coordinate();

        obj = viewer->insertPointSet(coord.size(), &coord[0][0], color,
                                     componentsPerColor);
    }

    if (d_color.get())
        d_color.get()->clearModified();
    if (d_coord.get())
        d_coord.get()->clearModified();

    return obj;
}
