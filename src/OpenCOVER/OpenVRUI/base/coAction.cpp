/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/sginterface/vruiIntersection.h>

#include <OpenVRUI/sginterface/vruiActionUserData.h>
#include <OpenVRUI/sginterface/vruiNode.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

coTouchAction::coTouchAction()
{
    actionName = "coTouchAction";
}

coTouchAction::~coTouchAction()
{
}

coAction::coAction()
{
    thisFrame = 0;
    child = 0;
    parent = 0;
    myNode = 0;
    userData = vruiRendererInterface::the()->createActionUserData(this);
    actionName = "coAction";
}

coAction::~coAction()
{
    if (userData)
        userData->action = 0;
    // in einer der Listen koennte das drin sein
    // wenn nicht ist auch nicht schlimm
    // oh doch, wenn es sie garnicht gibt!
    vruiIntersection *inters = vruiIntersection::getIntersectorForAction("coAction");
    if (inters)
        inters->remove(this);
    inters = vruiIntersection::getIntersectorForAction("coTouchAction");
    if (inters)
        inters->remove(this);

    if (myNode) // if we are attached to a node, make sure we are removed
    {
        vruiActionUserData *actionData = dynamic_cast<vruiActionUserData *>(myNode->getUserData(actionName));
        if (actionData)
        {
            if (actionData->action == this)
            {
                if (parent)
                    myNode->setUserData(actionName, parent->userData);
                else
                    myNode->setUserData(actionName, child->userData);
            }
        }
    }

    if (parent)
    {
        parent->child = child;
    }

    if (child)
    {
        child->parent = parent;
    }

    vruiRendererInterface::the()->deleteUserData(userData);
}

// miss is called once after a hit, if the node is not intersected
// anymore
void coAction::miss()
{
    VRUILOG("coAction::miss err: base class called!, do not return ACTION_CALL_ON_MISS if you don't have a miss() method or implement one!!")
}

int coAction::hitAll(vruiHit *hit)
{

    coAction *tmp = this;
    int ret = 0;
    while (tmp)
    {
        ret |= tmp->hit(hit);
        tmp = tmp->child;
    }
    return ret;
}

void coAction::missAll()
{
    coAction *tmp = this;
    while (tmp)
    {
        tmp->miss();
        tmp = tmp->child;
    }
}

void coAction::addChild(coAction *child)
{

    if (child && this != child)
    {
        coAction *tmp = this;
        while (tmp->parent)
            tmp = tmp->parent;

        while (tmp->child)
        {
            if (tmp == child)
                return;
            tmp = tmp->child;
        }

        coAction *oldchild = this->child;
        this->child = child;
        tmp = child;

        while (tmp->parent)
            tmp = tmp->parent;

        tmp->parent = this;

        tmp = child;

        while (tmp->child)
            tmp = tmp->child;

        tmp->child = oldchild;
    }
}

void coAction::setNode(vruiNode *node)
{
    if (node)
    {

        vruiActionUserData *actionData = dynamic_cast<vruiActionUserData *>(node->getUserData(actionName));

        if ((actionData == 0) || (actionData->action == 0))
        {
            node->setUserData(actionName, userData);
        }
        else
        {
            actionData->action->addChild(this);
        }
    }
    else
    {
        if (myNode)
        {
            vruiActionUserData *actionData = dynamic_cast<vruiActionUserData *>(myNode->getUserData(actionName));
            if (actionData)
            {
                if (actionData->action == this)
                {
                    if (parent)
                    {
                        myNode->setUserData(actionName, parent->userData);
                    }
                    else
                    {
                        if (child)
                            myNode->setUserData(actionName, child->userData);
                        else
                            myNode->setUserData(actionName, 0);
                    }
                }
            }
        }
    }

    myNode = node;
}
}
