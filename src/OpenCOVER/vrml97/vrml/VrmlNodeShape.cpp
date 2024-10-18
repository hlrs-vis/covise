/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeShape.cpp

#include "VrmlNode.h"

#include "VrmlNodeAppearance.h"
#include "VrmlNodeGeometry.h"
#include "VrmlNodeTexture.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

#include "VrmlNodeShape.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeShape(s); }

// Define the built in VrmlNodeType:: "Shape" fields

void VrmlNodeShape::initFields(VrmlNodeShape *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("appearance", node->d_appearance),
                     exposedField("geometry", node->d_geometry),
                     exposedField("effect", node->d_effect));
}

const char *VrmlNodeShape::name() { return "Shape"; }


VrmlNodeShape::VrmlNodeShape(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_viewerObject(0)
{
}

void VrmlNodeShape::cloneChildren(VrmlNamespace *ns)
{
    if (d_appearance.get())
    {
        d_appearance.set(d_appearance.get()->clone(ns));
        d_appearance.get()->parentList.push_back(this);
    }
    if (d_geometry.get())
    {
        d_geometry.set(d_geometry.get()->clone(ns));
        d_geometry.get()->parentList.push_back(this);
    }
}

bool VrmlNodeShape::isModified() const
{
    return (d_modified
            || (d_geometry.get() && d_geometry.get()->isModified())
            || (d_appearance.get() && d_appearance.get()->isModified())
            || (d_effect.get() && d_effect.get()->isModified()));
}

void VrmlNodeShape::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_appearance.get())
        d_appearance.get()->clearFlags();
    if (d_geometry.get())
        d_geometry.get()->clearFlags();
    if (d_effect.get())
        d_effect.get()->clearFlags();
}

void VrmlNodeShape::addToScene(VrmlScene *s, const char *relUrl)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_appearance.get())
        d_appearance.get()->addToScene(s, relUrl);
    if (d_geometry.get())
        d_geometry.get()->addToScene(s, relUrl);
    if (d_effect.get())
        d_effect.get()->addToScene(s, relUrl);
    nodeStack.pop_front();
}

void VrmlNodeShape::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_appearance.get())
        d_appearance.get()->copyRoutes(ns);
    if (d_geometry.get())
        d_geometry.get()->copyRoutes(ns);
    if (d_effect.get())
        d_effect.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

VrmlNodeShape *VrmlNodeShape::toShape() const
{
    return (VrmlNodeShape *)this;
}

void VrmlNodeShape::render(Viewer *viewer)
{
    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }

    VrmlNodeGeometry *g = d_geometry.get() ? d_geometry.get()->toGeometry() : 0;

    if (d_viewerObject)
        viewer->insertReference(d_viewerObject);

    else if (g)
    {
        d_viewerObject = viewer->beginObject(name(), 0, this);

        // Don't care what color it is if we are picking
        bool picking = (Viewer::RENDER_MODE_PICK == viewer->getRenderMode());
        if (!picking)
        {
            int nTexComponents = 0;

            if (!picking && d_appearance.get() && d_appearance.get()->toAppearance())
            {
                VrmlNodeAppearance *a = d_appearance.get()->toAppearance();
                a->render(viewer);

                if (a->texture() && a->texture()->toTexture())
                    nTexComponents = a->texture()->toTexture()->nComponents();
            }
            else
            {
                viewer->setColor(1.0, 1.0, 1.0); // default object color
                viewer->enableLighting(false); // turn lighting off
            }

            // hack for opengl material mode
            viewer->setMaterialMode(nTexComponents, g->color() != 0);
        }

        // render geometry
        g->render(viewer);

        if (d_effect.get())
        {
            VrmlNode *e = d_effect.get();
            e->render(viewer);
        }
        viewer->endObject();
    }

    else
    {
        if (d_appearance.get())
            d_appearance.get()->clearModified();
        if (d_effect.get())
            d_effect.get()->clearModified();
    }

    clearModified();
}

bool VrmlNodeShape::isOnlyGeometry() const
{
    if (!VrmlNodeChild::isOnlyGeometry())
        return false;

    if (d_appearance.get() && !d_appearance.get()->isOnlyGeometry())
    {
        //std::cerr << "Nsa" << std::flush;
        return false;
    }
    if (d_geometry.get() && !d_geometry.get()->isOnlyGeometry())
    {
        //std::cerr << "Nga" << std::flush;
        return false;
    }
    if (d_effect.get() && !d_effect.get()->isOnlyGeometry())
    {
        //std::cerr << "Nea" << std::flush;
        return false;
    }

    return true;
}
