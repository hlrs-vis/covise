/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

using namespace std;

namespace vrui
{

coInteraction::coInteraction(InteractionType type, const string &name, InteractionPriority priority)
    : notifyOnly(false)
{
    //fprintf(stderr,"coInteraction::coInteraction \n");

    this->type = type;
    state = Idle;
    this->priority = priority;
    this->name = name;
    registered = false;
    hasPriorityFlag = false;

	runningState = StateNotRunning;
}

coInteraction::~coInteraction()
{
    //fprintf(stderr,"coInteraction::~coInteraction \n");
    if (state == Active || state == Paused || state == ActiveNotify)
        cancelInteraction();
    state = Idle;
    coInteractionManager::the()->unregisterInteraction(this);
}

void coInteraction::setGroup(coInteraction::InteractionGroup group)
{
    this->group = group;

}

void coInteraction::pause()
{
    //fprintf(stderr,"coInteraction::paus \n");
    state = Paused;
	coInteractionManager::the()->doRemoteUnLock(group);
}

void coInteraction::requestActivation()
{
    //fprintf(stderr,"coInteraction::requestActivation \n");
    if (state == Idle) // if we are already active or the remote i active, we don't try to become active
    {
        state = PendingActive;
    }
}

void coInteraction::update()
{
}
void coInteraction::resetState()
{
}

void coInteraction::cancelInteraction()
{
    //fprintf(stderr,"coInteraction::cancelInteraction \n");
    if (state == Active || state == Paused || state == ActiveNotify)
    {
        state = Idle;
		coInteractionManager::the()->doRemoteUnLock(group);
    }
}

bool coInteraction::activate()
{
    //fprintf(stderr,"coInteraction::activate \n");
    if (state == Active || state == Paused || state == ActiveNotify)
    {
        //fprintf(stderr,"coInteraction::activate (state == Active || state == Paused || state == ActiveNotify)\n");
        return false;
    }
    else if (state == Idle)
    {
        //fprintf(stderr,"coInteraction::activate (state == Idle)\n");

        if (coInteractionManager::the()->isOneActive(type))
        {
			cerr << "interaction " << name << " of type " << type << " is blocked" << std::endl;
			return false;
        }
        else if (group != GroupNonexclusive && coInteractionManager::the()->isOneActive(group))
        {
			cerr << "interaction " << name << " of group " << group << " is remote blocked" << std::endl;
			return false;
        }
		else if (isNotifyOnly())
		{
			state = ActiveNotify;
			return true;
		}
		else
		{
			state = Active;
			coInteractionManager::the()->doRemoteLock(group);
			return true;
		}
    }
    return false;
}

void coInteraction::cancelPendingActivation()
{
    //fprintf(stderr,"coInteraction::cancelPendingActivatio \n");
    state = Idle;
}

void coInteraction::doActivation()
{
    //fprintf(stderr,"coInteraction::doActivation \n");
    state = notifyOnly ? ActiveNotify : Active;
}

void coInteraction::removeIcon() // remove the indicator for this interaction
{
    //fprintf(stderr,"coInteraction::removeIcon \n");
    hasPriorityFlag = false;
    if (type == ButtonA || type == AllButtons)
    {
        vruiRendererInterface::the()->removePointerIcon(name);
    }
}

void coInteraction::addIcon() // add    the indicator for this interaction
{
    //fprintf(stderr,"coInteraction::addIcon \n");
    hasPriorityFlag = true;

    if (type == ButtonA || type == AllButtons)
    {
        vruiRendererInterface::the()->addPointerIcon(name);
    }
}

bool coInteraction::hasPriority()
{
    //fprintf(stderr,"coInteraction::hasPriority \n");
    return hasPriorityFlag;
}

void coInteraction::setName(const string &name)
{
    //fprintf(stderr,"coInteraction::setName \n");
    bool haveToAddIcon = false;
    if (hasPriority())
    {
        haveToAddIcon = true;
        removeIcon();
    }

    this->name = name;

    if (haveToAddIcon)
    {
        addIcon();
    }
}

void coInteraction::setNotifyOnly(bool flag)
{
    //fprintf(stderr,"coInteraction::setNotifyOnly \n");
    notifyOnly = flag;
}

coInteraction::InteractionState coInteraction::getState()
{
	return state;
}

void coInteraction::setState(InteractionState s)
{
	if (state == InteractionState::Idle && s != InteractionState::Idle)
	{
		coInteractionManager::the()->doRemoteLock(group);
	}
	else
	{
		coInteractionManager::the()->doRemoteUnLock(group);
	}
	state = s;
}
}
