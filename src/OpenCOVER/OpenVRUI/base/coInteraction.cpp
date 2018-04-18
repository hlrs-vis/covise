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
    remoteLockID = 0; // don't synchronize Interaction
    remoteLock = false;

    runningState = StateNotRunning;
}

coInteraction::~coInteraction()
{
    //fprintf(stderr,"coInteraction::~coInteraction \n");
    if (state == Active || state == Paused || state == ActiveNotify)
        cancelInteraction();
    if ((remoteLock) && (remoteLockID > 0) && vruiRendererInterface::the()->isLockedByMe(remoteLockID))
    {
        vruiRendererInterface::the()->remoteUnLock(remoteLockID);
    }
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
}
void coInteraction::setRemoteLockID(int ID)
{
    //fprintf(stderr,"coInteraction::setRemoteLockID \n");
    remoteLockID = ID;
}

void coInteraction::setRemoteLock(bool s)
{
    //fprintf(stderr,"coInteraction::setRemoteLock \n");
    remoteLock = s;
    if ((remoteLock) && (remoteLockID > 0)) // remote locking has been enabled
    {
        if (state == Active || state == ActiveNotify)
        {
            if (vruiRendererInterface::the()->isLocked(remoteLockID) && !vruiRendererInterface::the()->isLockedByMe(remoteLockID))
            {
                // someone else has a lock on this interaction so pause our own
                state = Paused;
            }
            else if (!vruiRendererInterface::the()->isLockedByMe(remoteLockID))
            {
                vruiRendererInterface::the()->remoteLock(remoteLockID);
            }
        }
    }
    else // remote locking has been disabled
    {
        if (state == RemoteActive)
        {
            state = Idle;
        }
        if (vruiRendererInterface::the()->isLockedByMe(remoteLockID))
        {
            vruiRendererInterface::the()->remoteUnLock(remoteLockID);
        }
    }
}

void coInteraction::setRemoteActive(bool ra)
{
    //fprintf(stderr,"coInteraction::setRemoteActive \n");
    if (ra)
    {
        if (state == Active || state == ActiveNotify)
        {
            if (vruiRendererInterface::the()->isLocked(remoteLockID) && !vruiRendererInterface::the()->isLockedByMe(remoteLockID))
            {
                // someone else has a lock on this interaction so pause our own
                state = Paused;
            }
        }
        state = RemoteActive;
    }
    else if (state == RemoteActive)
    {
        state = Idle;
    }
    else
    {
        // other cases should never happen
        cerr << "this should not happen, remoteActive false without being true" << endl;
    }
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
        if (isNotifyOnly())
        {
            if ((remoteLock) && (remoteLockID > 0) && (vruiRendererInterface::the()->isLocked(remoteLockID) && !vruiRendererInterface::the()->isLockedByMe(remoteLockID)))
            {
                state = RemoteActive;
                return false;
            }
            state = ActiveNotify;
            return true;
        }
        else if (coInteractionManager::the()->isOneActive(type))
        {
            return false;
        }
        else if (group != GroupNonexclusive && coInteractionManager::the()->isOneActive(group))
        {
            return false;
        }
        else
        {
            if ((remoteLock) && (remoteLockID > 0) && (vruiRendererInterface::the()->isLocked(remoteLockID) && !vruiRendererInterface::the()->isLockedByMe(remoteLockID)))
            {
                state = RemoteActive;
                return false;
            }
            state = Active;
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
    if ((remoteLock) && (remoteLockID > 0) && (vruiRendererInterface::the()->isLocked(remoteLockID) && !vruiRendererInterface::the()->isLockedByMe(remoteLockID)))
    {
        state = RemoteActive;
        return;
    }
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
}
