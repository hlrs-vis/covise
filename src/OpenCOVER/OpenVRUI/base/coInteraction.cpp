/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

using namespace std;

#define INTERACTOR_DEBUG false

#define DEBUG_OUTPUT(text) \
    do \
    { \
        if (INTERACTOR_DEBUG) \
        { \
            fprintf(stderr, "%s\n", text); \
        } \
    } while (false)

namespace vrui
{

coInteraction::coInteraction(InteractionType type, const string &name, InteractionPriority priority)
    : notifyOnly(false)
{
    DEBUG_OUTPUT("coInteraction::coInteraction");

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
    DEBUG_OUTPUT("coInteraction::~coInteraction");
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
    DEBUG_OUTPUT("coInteraction::pause");
    state = Paused;
    if(group == GroupNavigation)
	    coInteractionManager::the()->doRemoteUnLock();
}

void coInteraction::requestActivation()
{
    DEBUG_OUTPUT("coInteraction::requestActivation");
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
    DEBUG_OUTPUT("coInteraction::cancelInteraction");
    if (state == Active || state == Paused || state == ActiveNotify)
    {
        state = Idle;
        if(group == GroupNavigation)
		    coInteractionManager::the()->doRemoteUnLock();
    }
}

bool coInteraction::activate()
{
    DEBUG_OUTPUT("coInteraction::activate");
    if (state == Active || state == Paused || state == ActiveNotify)
    {
        DEBUG_OUTPUT("coInteraction::activate (state == Active || state == Paused || state == ActiveNotify)\n");
        return false;
    }
    else if (state == Idle)
    {
        DEBUG_OUTPUT("coInteraction::activate (state == Idle)\n");

        if (coInteractionManager::the()->isOneActive(type))
        {
			return false;
        }
        else if (group == GroupNavigation && coInteractionManager::the()->isOneActive(group))
        {
            return false;
        }
        else if (isNotifyOnly())
		{
			state = ActiveNotify;
			return true;
		}
        else{
            state = Active;
            coInteractionManager::the()->doRemoteLock();
            return true;
        }
    }
    return false;
}

void coInteraction::cancelPendingActivation()
{
    DEBUG_OUTPUT("coInteraction::cancelPendingActivatio");
    state = Idle;
}

void coInteraction::doActivation()
{
    DEBUG_OUTPUT("coInteraction::doActivation");
    state = notifyOnly ? ActiveNotify : Active;
}

void coInteraction::removeIcon() // remove the indicator for this interaction
{
    DEBUG_OUTPUT("coInteraction::removeIcon");
    hasPriorityFlag = false;
    if (type == ButtonA || type == AllButtons)
    {
        vruiRendererInterface::the()->removePointerIcon(name);
    }
}

void coInteraction::addIcon() // add    the indicator for this interaction
{
    DEBUG_OUTPUT("coInteraction::addIcon");
    hasPriorityFlag = true;

    if (type == ButtonA || type == AllButtons)
    {
        vruiRendererInterface::the()->addPointerIcon(name);
    }
}

bool coInteraction::hasPriority()
{
    DEBUG_OUTPUT("coInteraction::hasPriority");
    return hasPriorityFlag;
}

void coInteraction::setName(const string &name)
{
    DEBUG_OUTPUT("coInteraction::setName");
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
    DEBUG_OUTPUT("coInteraction::setNotifyOnly");
    notifyOnly = flag;
}

coInteraction::InteractionState coInteraction::getState()
{
	return state;
}

void coInteraction::setState(InteractionState s)
{
	if(group == InteractionGroup::GroupNavigation)
    {
        if (state == InteractionState::Idle && s != InteractionState::Idle)
            coInteractionManager::the()->doRemoteLock();
        else
            coInteractionManager::the()->doRemoteUnLock();
    }
	state = s;
}
}
