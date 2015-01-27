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

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeCollision(s); }

// Define the built in VrmlNodeType:: "Collision" fields

VrmlNodeType *VrmlNodeCollision::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Collision", creator);
    }

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("collide", VrmlField::SFBOOL);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addField("proxy", VrmlField::SFNODE);
    t->addEventOut("collideTime", VrmlField::SFTIME);

    return t;
}

VrmlNodeType *VrmlNodeCollision::nodeType() const { return defineType(0); }

VrmlNodeCollision::VrmlNodeCollision(VrmlScene *scene)
    : VrmlNodeGroup(scene)
    , d_collide(true)
{
}

VrmlNodeCollision::~VrmlNodeCollision()
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
            if ((kid->toTouchSensor() && kid->toTouchSensor()->isEnabled()) || (kid->toPlaneSensor() && kid->toPlaneSensor()->isEnabled()) || (kid->toCylinderSensor() && kid->toCylinderSensor()->isEnabled()) || (kid->toSphereSensor() && kid->toSphereSensor()->isEnabled()) || (kid->toSpaceSensor() && kid->toSpaceSensor()->isEnabled()))
            {
                if (++nSensors == 1)
                    viewer->setSensitive(this);
            }
        }

        // Do the rest of the children (except the scene-level lights)
        for (i = 0; i < n; ++i)
            if (!(/*d_children[i]->toLight() ||*/
                  d_children[i]->toPlaneSensor() || d_children[i]->toSpaceSensor() || d_children[i]->toTouchSensor()))
                d_children[i]->render(viewer);

        // Turn off sensitivity
        if (nSensors > 0)
            viewer->setSensitive(0);

        viewer->setCollision(d_collide.get());

        viewer->endObject();
    }

    clearModified();
}

VrmlNode *VrmlNodeCollision::cloneMe() const
{
    return new VrmlNodeCollision(*this);
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

std::ostream &VrmlNodeCollision::printFields(std::ostream &os, int indent)
{
    if (!d_collide.get())
        PRINT_FIELD(collide);
    if (d_proxy.get())
        PRINT_FIELD(proxy);

    VrmlNodeGroup::printFields(os, indent);
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCollision::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    // check against both fieldnames cause scene() and getNamespace() maybe NULL
    // no easy way to check X3D status
    if ((strcmp(fieldName, "collide") == 0) || (strcmp(fieldName, "enabled") == 0))
    {
        if (fieldValue.toSFBool())
            d_collide = (VrmlSFBool &)fieldValue;
        else
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s).\n",
                               fieldValue.fieldTypeName(), "collide or enabled", nodeType()->getName(), "SFBool");
    }
    else if
        TRY_SFNODE_FIELD(proxy, Child)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeCollision::getField(const char *fieldName) const
{
    // check against both fieldnames cause scene() and getNamespace() maybe NULL
    // no easy way to check X3D status
    if ((strcmp(fieldName, "collide") == 0) || (strcmp(fieldName, "enabled") == 0))
        return &d_collide;
    else if (strcmp(fieldName, "proxy") == 0)
        return &d_proxy;
    else if (strcmp(fieldName, "collideTime") == 0)
        return &d_collideTime;

    return VrmlNodeGroup::getField(fieldName);
}
