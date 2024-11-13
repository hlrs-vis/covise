/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//
//  VrmlNodeGroup.cpp
//

#include "VrmlNodeGroup.h"

#include "VrmlNodeType.h"

#include "VrmlNodeProto.h"
#include "VrmlNodePlaneSensor.h"
#include "VrmlNodeSpaceSensor.h"
#include "VrmlNodeTouchSensor.h"
#include "VrmlNodeSphereSensor.h"
#include "VrmlNodeCylinderSensor.h"

#include "VrmlMFNode.h"
#include "VrmlSFVec3f.h"
#include "VrmlNodeGeometry.h"
#include "VrmlNodeProto.h"
#include "VrmlNodeShape.h"

#include "System.h"

using std::cerr;
using std::endl;
using namespace vrml;

// Return a new VrmlNodeGroup
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeGroup(s); }

// Define the built in VrmlNodeType:: "Group" fields

void VrmlNodeGroup::initFields(VrmlNodeGroup *node, VrmlNodeType *t) {
    initFieldsHelper(node, t,
        exposedField("children", node->d_children, [node](auto children){
            node->childrenChanged();
        }),
        field("bboxCenter", node->d_bboxCenter),
        field("bboxSize", node->d_bboxSize)
    );
    if(t)
    {
        t->addEventIn("addChildren", VrmlField::MFNODE);
        t->addEventIn("removeChildren", VrmlField::MFNODE);
    }
    VrmlNodeChild::initFields(node, t);
}


VrmlNodeGroup::VrmlNodeGroup(VrmlScene *scene, const std::string &name)
    : VrmlNodeChild(scene, name == "" ? this->name() : name)
    , d_bboxSize(-1.0, -1.0, -1.0)
    , d_parentTransform(0)
    , d_viewerObject(0)
{
}

VrmlNodeGroup::~VrmlNodeGroup()
{
    // delete d_viewerObject...
    while (d_children.size())
    {
        // don't, this causes an endless loop if(d_children[0])
        //{
        d_children.removeNode(d_children[0]);
        //}
    }
}

void VrmlNodeGroup::flushRemoveList()
{
    while (d_childrenToRemove.size())
    {
        d_childrenToRemove.removeNode(d_childrenToRemove[0]);
    }
}

void VrmlNodeGroup::cloneChildren(VrmlNamespace *ns)
{
    int n = d_children.size();
    VrmlNode **kids = d_children.get();
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

bool VrmlNodeGroup::isModified() const
{
    if (d_modified)
        return true;

    int n = d_children.size();

    for (int i = 0; i < n; ++i)
    {
        if (d_children[i] == NULL)
            return false;
        if (d_children[i]->isModified())
            return true;
    }

    return false;
}

void VrmlNodeGroup::clearFlags()
{
    VrmlNode::clearFlags();

    int n = d_children.size();
    for (int i = 0; i < n; ++i)
        if (d_children[i])
            d_children[i]->clearFlags();
}

void VrmlNodeGroup::addToScene(VrmlScene *s, const char *relativeUrl)
{
    d_scene = s;

    nodeStack.push_front(this);
    System::the->debug("VrmlNodeGroup::addToScene( %s )\n",
                       relativeUrl ? relativeUrl : "<null>");

    const char *currentRel = d_relative.get();
    if (!currentRel || !relativeUrl || strcmp(currentRel, relativeUrl) != 0)
        d_relative.set(relativeUrl);

    int n = d_children.size();

    for (int i = 0; i < n; ++i)
    {
        if (d_children[i])
            d_children[i]->addToScene(s, d_relative.get());
    }
    nodeStack.pop_front();
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeGroup::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns); // Copy my routes

    // Copy childrens' routes
    int n = d_children.size();
    for (int i = 0; i < n; ++i)
        if (d_children[i])
            d_children[i]->copyRoutes(ns);
    nodeStack.pop_front();
}

VrmlNode *VrmlNodeGroup::getParentTransform() { return d_parentTransform; }

bool VrmlNodeGroup::isOnlyGeometry() const
{
    if (!VrmlNode::isOnlyGeometry())
        return false;

    int n = d_children.size();
    for (int i=0; i<n; ++i)
    {
        if (!d_children[i]->isOnlyGeometry())
        {
            //std::cerr << "Nc" << i << std::flush;
            return false;
        }
    }

    return true;
}

void VrmlNodeGroup::checkAndRemoveNodes(Viewer *viewer)
{
    if (d_childrenToRemove.size())
    {
        viewer->beginObject(name(), 0, this);
        int i, n = d_childrenToRemove.size();
        for (i = 0; i < n; i++)
        {
            Viewer::Object child_viewerObject = 0;
            VrmlNode *kid = d_childrenToRemove[i];
            if (kid->as<VrmlNodeGeometry>())
                child_viewerObject = kid->as<VrmlNodeGeometry>()->getViewerObject();
            else if (kid->as<VrmlNodeGroup>())
                child_viewerObject = kid->as<VrmlNodeGroup>()->d_viewerObject;
            else if (kid->as<VrmlNodeProto>())
                child_viewerObject = kid->as<VrmlNodeProto>()->getViewerObject();
            else if (kid->as<VrmlNodeShape>())
                child_viewerObject = kid->as<VrmlNodeShape>()->getViewerObject();
            if (child_viewerObject)
                viewer->removeChild(child_viewerObject);
        }
        viewer->endObject();
    }
    while (d_childrenToRemove.size())
    {
        d_childrenToRemove.removeNode(d_childrenToRemove[0]);
    }
}
// Render each of the children

void VrmlNodeGroup::render(Viewer *viewer)
{
    if (!haveToRender())
    {
        return;
    }

    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }
    checkAndRemoveNodes(viewer);

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
            if (kid && ((kid->as<VrmlNodeTouchSensor>() && kid->as<VrmlNodeTouchSensor>()->isEnabled()) || (kid->as<VrmlNodePlaneSensor>() && kid->as<VrmlNodePlaneSensor>()->isEnabled()) || (kid->as<VrmlNodeCylinderSensor>() && kid->as<VrmlNodeCylinderSensor>()->isEnabled()) || (kid->as<VrmlNodeSphereSensor>() && kid->as<VrmlNodeSphereSensor>()->isEnabled()) || (kid->as<VrmlNodeSpaceSensor>() && kid->as<VrmlNodeSpaceSensor>()->isEnabled())))
            {
                if (++nSensors == 1)
                    viewer->setSensitive(this);
            }
        }

        // Do the rest of the children (except the scene-level lights)
        for (i = 0; i < n; ++i)
        {
            if (d_children[i])
            {
                if (!(/*d_children[i]->toLight() ||*/
                      d_children[i]->as<VrmlNodePlaneSensor>() || d_children[i]->as<VrmlNodeCylinderSensor>() || d_children[i]->as<VrmlNodeSpaceSensor>() || d_children[i]->as<VrmlNodeTouchSensor>()))
                    d_children[i]->render(viewer);
            }
        }

        // Turn off sensitivity
        if (nSensors > 0)
            viewer->setSensitive(0);

        viewer->endObject();
    }

    clearModified();
}

// Accumulate transforms
// Cache a pointer to (one of the) parent transforms for proper
// rendering of bindables.

void VrmlNodeGroup::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;

    int i, n = d_children.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_children[i];
        if (kid)
            kid->accumulateTransform(parent);
    }
}

// Pass on to enabled touchsensor child.

void VrmlNodeGroup::activate(double time,
                             bool isOver, bool isActive,
                             double *p, double *M)
{
    int i, n = d_children.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_children[i];

        if (kid == NULL)
            continue;
        if (kid->as<VrmlNodeTouchSensor>() && kid->as<VrmlNodeTouchSensor>()->isEnabled())
        {
            kid->as<VrmlNodeTouchSensor>()->activate(time, isOver, isActive, p);
            break;
        }
        else if (kid->as<VrmlNodePlaneSensor>() && kid->as<VrmlNodePlaneSensor>()->isEnabled())
        {
            kid->as<VrmlNodePlaneSensor>()->activate(time, isActive, p);
            break;
        }
        else if (kid->as<VrmlNodeSpaceSensor>() && kid->as<VrmlNodeSpaceSensor>()->isEnabled())
        {
            kid->as<VrmlNodeSpaceSensor>()->activate(time, isActive, p, M);
            break;
        }
        else if (kid->as<VrmlNodeSphereSensor>() && kid->as<VrmlNodeSphereSensor>()->isEnabled())
        {
            kid->as<VrmlNodeSphereSensor>()->activate(time, isActive, p);
            break;
        }
        else if (kid->as<VrmlNodeCylinderSensor>() && kid->as<VrmlNodeCylinderSensor>()->isEnabled())
        {
            kid->as<VrmlNodeCylinderSensor>()->activate(time, isActive, p);
            break;
        }
    }
}

void VrmlNodeGroup::addChildren(const VrmlMFNode &children)
{
    int nNow = d_children.size();
    int n = children.size();

    for (int i = 0; i < n; ++i)
    {
        VrmlNode *child = children[i];
        if (child == NULL)
        {
            continue;
        }

        child->parentList.push_back(this);
        if (child->getTraversalForce() > 0)
        {
            forceTraversal(false, child->getTraversalForce());
        }

        VrmlNodeProto *p = 0;

        // Add legal children and un-instantiated EXTERNPROTOs
        // Is it legal to add null children nodes?
        if (child == 0 || child->as<VrmlNodeChild>() || ((p = child->as<VrmlNodeProto>()) != 0 && p->size() == 0))
        {
            d_children.addNode(child);
            if (child)
            {
                child->addToScene(d_scene, d_relative.get());
                child->accumulateTransform(d_parentTransform);
            }
        }
        else
            System::the->error("Error: Attempt to add a %s node as a child of a %s node.\n",
                               child->nodeType()->getName(), nodeType()->getName());
    }

    if (nNow != d_children.size())
    {
        //??eventOut( d_scene->timeNow(), "children_changed", d_children );
        setModified();
    }
}

void VrmlNodeGroup::removeChildren(const VrmlMFNode &children)
{
    int nNow = d_children.size();
    int n = children.size();

    for (int i = 0; i < n; ++i)
    {
        if (children[i])
        {
            children[i]->decreaseTraversalForce();
            children[i]->parentList.remove(this);
        }
        else
        {
            cerr << "VrmlNodeGroup::removeChildren1: NULL child" << endl;
        }
        d_childrenToRemove.addNode(children[i]);
        d_children.removeNode(children[i]);
    }

    if (nNow != d_children.size())
    {
        //??eventOut( d_scene->timeNow(), "children_changed", d_children );
        setModified();
    }
}

void VrmlNodeGroup::removeChildren()
{
    int n = d_children.size();

    for (int i = n; i > 0; --i)
    {
        if (d_children[i - 1])
        {
            d_children[i - 1]->decreaseTraversalForce();
            d_children[i - 1]->parentList.remove(this);
        }
        else
        {
            cerr << "VrmlNodeGroup::removeChildren2: NULL child" << endl;
        }
        d_childrenToRemove.addNode(d_children[i - 1]);
        d_children.removeNode(d_children[i - 1]);
    }

    setModified();
}

void VrmlNodeGroup::eventIn(double timeStamp,
                            const char *eventName,
                            const VrmlField *fieldValue)
{
    if (!fieldValue)
        return;

    if (strcmp(eventName, "addChildren") == 0)
    {
        if (fieldValue->toMFNode()) // check that fieldValue is MFNode
            addChildren(*(fieldValue->toMFNode()));
        else
            System::the->error("VrmlNodeGroup.%s %s eventIn invalid field type.\n",
                               name(), eventName);
    }

    else if (strcmp(eventName, "removeChildren") == 0)
    {
        if (fieldValue->toMFNode()) // check that fieldValue is MFNode
            removeChildren(*(fieldValue->toMFNode()));
        else
            System::the->error("VrmlNodeGroup.%s %s eventIn invalid field type.\n",
                               name(), eventName);
    }

    else if ((strcmp(eventName, "children") == 0) || (strcmp(eventName, "set_children") == 0))
    {
        removeChildren();
        if (fieldValue->toMFNode()) // check that fieldValue is MFNode
            addChildren(*(fieldValue->toMFNode()));
        else
            System::the->error("VrmlNodeGroup.%s %s eventIn invalid field type.\n",
                               name(), eventName);
    }

    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

// Set the value of one of the node fields.
void VrmlNodeGroup::updateChildren()
{
    for (int i = 0; i < d_oldChildren.size(); i++)
    {
        if (d_children[i])
        {
            d_children[i]->decreaseTraversalForce();
            d_children[i]->parentList.remove(this);
        }
        else
        {
            cerr << "VrmlNodeGroup::setField(children): had NULL child" << endl;
        }
    }


    d_oldChildren = d_children;

    for (int i = 0; i < d_children.size(); i++)
    {
        VrmlNode *child = d_children[i];
        if (child == NULL)
        {
            continue;
        }

        child->parentList.push_back(this);
        if (child->getTraversalForce() > 0)
        {
            forceTraversal(false, child->getTraversalForce());
        }
    }
}

const VrmlField *VrmlNodeGroup::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "bboxCenter") == 0)
        return &d_bboxCenter;
    else if (strcmp(fieldName, "bboxSize") == 0)
        return &d_bboxSize;
    else if (strcmp(fieldName, "children") == 0)
        return &d_children;

    return VrmlNodeChild::getField(fieldName);
}

int VrmlNodeGroup::size()
{
    return d_children.size();
}

VrmlNode *VrmlNodeGroup::child(int index)
{
    if (index >= 0 && index < d_children.size())
        return d_children[index];

    return 0;
}

void VrmlNodeGroup::childrenChanged(  )
{
    //optional implementation in subclasses
}

