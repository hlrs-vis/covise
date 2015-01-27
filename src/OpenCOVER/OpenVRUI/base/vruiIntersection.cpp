/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/sginterface/vruiIntersection.h>

#include <OpenVRUI/sginterface/vruiActionUserData.h>
#include <OpenVRUI/sginterface/vruiNode.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

vector<int *> &vruiIntersection::frames()
{
    static vector<int *> frames;
    return frames;
}

vector<vruiIntersection *> &vruiIntersection::intersectors()
{
    static vector<vruiIntersection *> intersectors;
    return intersectors;
}

vruiIntersection::vruiIntersection()
{
    vruiRendererInterface::the()->getUpdateManager()->add(this);
    frameIndex = -1;
}

vruiIntersection::~vruiIntersection()
{
}

bool vruiIntersection::update()
{
    if (frameIndex < 0)
    {
        VRUILOG("vruiIntersection::update err: you have to set frameIndex in subclass " << getClassName())
        return false;
    }

    (*frames()[frameIndex])++;
    //VRUILOG("vruiIntersection::update info: frame " << (*frames[frameIndex]))
    intersect();
    callMisses();
    return true;
}

void vruiIntersection::add(vruiNode *node, coAction *action)
{
    action->setNode(node);
}

void vruiIntersection::remove(vruiNode *node)
{
    vruiActionUserData *actionData = dynamic_cast<vruiActionUserData *>(node->getUserData(getActionName()));

    if ((actionData) && (actionData->action))
    {
        actionData->action->setNode(0);
    }
}

void vruiIntersection::callActions(vruiNode *node, vruiHit *hit)
{

    if (frameIndex < 0)
    {
        VRUILOG("vruiIntersection::callActions err: you have to set frameIndex in subclass " << getClassName())
        return;
    }

    int thisFrame = (*frames()[frameIndex]);

    for (int ctr = 0; ctr < node->getNumParents(); ++ctr)
    {
        callActions(node->getParent(ctr), hit);
    }

    // if we don't have any userdata attached to this node, then nothing to do

    vruiActionUserData *actionData = dynamic_cast<vruiActionUserData *>(node->getUserData(getActionName()));

    if ((actionData) && (actionData->action))
    {
        int recall = actionData->action->hitAll(hit);
        if (recall & coAction::ACTION_CALL_ON_MISS)
        {
            if ((thisFrame != 0) && (actionData->action->getFrame() != (thisFrame - 1)))
            {
                actionList.push_back(actionData->action);
            }
            actionData->action->setFrame(thisFrame);
        }
        else
        {
            actionData->action->setFrame(thisFrame + 10);
        }
    }
}

void vruiIntersection::callMisses()
{

    if (frameIndex < 0)
    {
        VRUILOG("vruiIntersection::callActions err: you have to set frameIndex in subclass " << getClassName())
        return;
    }

    int thisFrame = (*frames()[frameIndex]);

    for (list<coAction *>::iterator action = actionList.begin(); action != actionList.end();)
    {
        if ((thisFrame != 0) && ((*action)->getFrame() != thisFrame))
        {
            (*action)->missAll();
            action = actionList.erase(action);
        }
        else
        {
            ++action;
        }
    }
}

void vruiIntersection::remove(coAction *action)
{
    actionList.remove(action);
    action->setNode(0);
}

vruiIntersection *vruiIntersection::getIntersectorForAction(const string &actionName)
{

    //VRUILOG("vruiIntersection::getIntersectorForAction info: called")
    for (unsigned int ctr = 0; ctr < intersectors().size(); ++ctr)
    {
        //VRUILOG("vruiIntersection::getIntersectorForAction info: trying " << intersectors[ctr]->getActionName())
        if (intersectors()[ctr]->getActionName() == actionName)
        {
            //VRUILOG("vruiIntersection::getIntersectorForAction info: found intersector [" << ctr << "]")
            return intersectors()[ctr];
        }
    }

    return 0;
}

vruiIntersection *vruiIntersection::getIntersector(const string &name)
{

    for (unsigned int ctr = 0; ctr < intersectors().size(); ++ctr)
    {
        if (intersectors()[ctr]->getClassName() == name)
        {
            //VRUILOG("vruiIntersection::getIntersector info: found intersector [" << ctr << "]")
            return intersectors()[ctr];
        }
    }

    return 0;
}
}
