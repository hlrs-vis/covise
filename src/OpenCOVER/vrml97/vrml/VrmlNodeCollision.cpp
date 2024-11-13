/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCollision.cpp

#include "VrmlNodeCollision.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"

#include "VrmlNodeProto.h"
#include "VrmlNodePlaneSensor.h"
#include "VrmlNodeSpaceSensor.h"
#include "VrmlNodeTouchSensor.h"
#include "VrmlNodeSphereSensor.h"
#include "VrmlNodeCylinderSensor.h"

using namespace vrml;


void VrmlNodeCollision::initFields(VrmlNodeCollision *node, VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("collide", node->d_collide),
                     exposedField("enabled", node->d_collide),
                     field("proxy", node->d_proxy));
    if (t)
        t->addEventOut("collideTime", VrmlField::SFTIME);                
                     
}

const char *VrmlNodeCollision::name() { return "Collision"; }

VrmlNodeCollision::VrmlNodeCollision(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
    , d_collide(true)
{
}

// Render each of the children

void VrmlNodeCollision::render(Viewer *viewer)
{
    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }

    if (d_viewerObject)
        viewer->insertReference(d_viewerObject);

    else if (d_children.size() > 0)
    {
        int i, n = d_children.size();
        int nSensors = 0;

        d_viewerObject = viewer->beginObject(name(), 0, this);

        // Draw nodes that impact their siblings (DirectionalLights,
        // TouchSensors, any others? ...)
        for (i = 0; i < n; ++i)
        {
            VrmlNode *kid = d_children[i];

            //if ( kid->toLight() ) && ! (kid->toPointLight() || kid->toSpotLight()) )
            //  kid->render(viewer);
            //else
            if ((kid->as<VrmlNodeTouchSensor>() && kid->as<VrmlNodeTouchSensor>()->isEnabled()) || (kid->as<VrmlNodePlaneSensor>() && kid->as<VrmlNodePlaneSensor>()->isEnabled()) || (kid->as<VrmlNodeCylinderSensor>() && kid->as<VrmlNodeCylinderSensor>()->isEnabled()) || (kid->as<VrmlNodeSphereSensor>() && kid->as<VrmlNodeSphereSensor>()->isEnabled()) || (kid->as<VrmlNodeSpaceSensor>() && kid->as<VrmlNodeSpaceSensor>()->isEnabled()))
            {
                if (++nSensors == 1)
                    viewer->setSensitive(this);
            }
        }

        // Do the rest of the children (except the scene-level lights)
        for (i = 0; i < n; ++i)
            if (!(/*d_children[i]->toLight() ||*/
                  d_children[i]->as<VrmlNodePlaneSensor>() || d_children[i]->as<VrmlNodeSpaceSensor>() || d_children[i]->as<VrmlNodeTouchSensor>()))
                d_children[i]->render(viewer);

        // Turn off sensitivity
        if (nSensors > 0)
            viewer->setSensitive(0);

        viewer->setCollision(d_collide.get());

        viewer->endObject();
    }

    clearModified();
}

void VrmlNodeCollision::cloneChildren(VrmlNamespace *ns)
{
    VrmlNodeGroup::cloneChildren(ns);
    if (d_proxy.get())
    {
        d_proxy.set(d_proxy.get()->clone(ns));
        d_proxy.get()->parentList.push_back(this);
    }
}

bool VrmlNodeCollision::isModified() const
{
    return ((d_proxy.get() && d_proxy.get()->isModified()) || VrmlNodeGroup::isModified());
}

void VrmlNodeCollision::clearFlags()
{
    VrmlNodeGroup::clearFlags();
    if (d_proxy.get())
        d_proxy.get()->clearFlags();
}

void VrmlNodeCollision::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    VrmlNodeGroup::addToScene(s, rel);
    if (d_proxy.get())
        d_proxy.get()->addToScene(s, rel);
    nodeStack.pop_front();
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeCollision::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNodeGroup::copyRoutes(ns);
    if (d_proxy.get())
        d_proxy.get()->copyRoutes(ns);
    nodeStack.pop_front();
}
