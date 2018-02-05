/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coInteractionManager.h>
#include <OpenVRUI/coButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

coInteractionManager *coInteractionManager::im = 0;

coInteractionManager::coInteractionManager()
{
    im = this;
}

coInteractionManager::~coInteractionManager()
{
}

coInteractionManager *coInteractionManager::the()
{
    return im;
}

bool coInteractionManager::isOneActive(coInteraction::InteractionType type)
{

    // the high priority elements are at the end of the list
    // therefore it is efficient to search from back to front
    // set pointer to dummy element behind the last element
    for (list<coInteraction *>::reverse_iterator it = interactionStack[type].rbegin();
         it != interactionStack[type].rend();
         ++it)
    {
        if ((*it)->getState() == coInteraction::Active)
        {
            return true;
        }
    }
    return false;
}

bool coInteractionManager::isOneActive(coInteraction::InteractionGroup group)
{
    for (int type = coInteraction::ButtonA; type < coInteraction::NumInteractorTypes; ++type)
    {
        // the high priority elements are at the end of the list
        // therefore it is efficient to search from back to front
        // set pointer to dummy element behind the last element
        for (list<coInteraction *>::reverse_iterator it = interactionStack[type].rbegin();
             it != interactionStack[type].rend();
             ++it)
        {
            if ((*it)->getState() == coInteraction::Active && (*it)->getGroup() == group)
            {
                return true;
            }
        }
    }
    return false;
}

bool coInteractionManager::update()
{

    //std::cerr << "coInteractionManager::update " << std::endl;

    bool change = false;

    // for all interactions types = buttons
    for (int i = 0; i < coInteraction::NumInteractorTypes; ++i)
    {
        //std::cerr << "coInteractionManager::update coInteraction::NumInteractorTypes " << i << std::endl;
        bool isActive = isOneActive((coInteraction::InteractionType)i);

        // go through the list of registerd interactions backwards
        // and check if an interaction wants to be active
        // if there is another active interaction, do not active the pending one
        // else activate it
        // update all interactions in the list
        for (list<coInteraction *>::reverse_iterator it = interactionStack[i].rbegin();
             it != interactionStack[i].rend();
             ++it)
        {
            //std::cerr << "coInteractionManager::update interactionStack[i].rbegin() " << i << std::endl;
            //std::cerr << "coInteractionManager::update " << i << "getState() " << (*it)->getState() << "getType() " << (*it)->getType()<< "getPriority() " << (*it)->getPriority() <<std::endl;

            if (((*it)->remoteLock) && ((*it)->remoteLockID > 0))
            {
                if ((vruiRendererInterface::the()->isLocked((*it)->remoteLockID) && !vruiRendererInterface::the()->isLockedByMe((*it)->remoteLockID)))
                { // this interaction is locked by a remote site
                    if ((*it)->getState() == coInteraction::PendingActive)
                    {
                        (*it)->cancelPendingActivation();
                        (*it)->setRemoteActive(true);
                        change = true;
                    }
                    if ((*it)->getState() == coInteraction::Active)
                    {
                        (*it)->cancelInteraction();
                        (*it)->pause();
                        change = true;
                    }
                }
                if ((*it)->getState() == coInteraction::Active)
                {
                    if (!vruiRendererInterface::the()->isLockedByMe((*it)->remoteLockID))
                    {
                        vruiRendererInterface::the()->remoteLock((*it)->remoteLockID);
                    }
                }
                if ((*it)->getState() == coInteraction::Idle)
                {
                    if (vruiRendererInterface::the()->isLockedByMe((*it)->remoteLockID))
                    {
                        vruiRendererInterface::the()->remoteUnLock((*it)->remoteLockID);
                    }
                }
                if ((*it)->getState() == coInteraction::Idle)
                {
                    (*it)->setRemoteActive(false);
                }
            }
            if ((*it)->getState() == coInteraction::PendingActive)
            {
                if ((*it)->isNotifyOnly())
                {
                    (*it)->doActivation();
                    change = true;
                }
                else if (isActive)
                {
                    (*it)->cancelPendingActivation();
                    change = true;
                }
                else
                {
                    (*it)->doActivation();
                    change = true;
                }
            }
            //std::cerr << "updating " << (*it)->getName() << std::endl;
            (*it)->update();

            if ((*it)->isRunning())
                change = true;
        }

        // go through the list of active but unregisted interactions
        // update all interactions in this list
        // if they are not running any more
        // remove it from this list
        for (list<coInteraction *>::iterator it = activeUnregisteredInteractions[i].begin();
             it != activeUnregisteredInteractions[i].end();)
        {
            if (*it)
            {
                (*it)->update();

                if ((*it)->getState() == coInteraction::Idle || (*it)->getState() == coInteraction::RemoteActive)
                {
                    if (((*it)->remoteLock) && ((*it)->remoteLockID > 0) && vruiRendererInterface::the()->isLockedByMe((*it)->remoteLockID))
                    {
                        vruiRendererInterface::the()->remoteUnLock((*it)->remoteLockID);
                    }
                    (*it)->resetState();
                    change = true;
                    it = activeUnregisteredInteractions[(*it)->getType()].erase(it);
                }
            }
            else
            {
                VRUILOG("coInteractionManager::update fixme: null interaction")
                (*it)->resetState();
                it = activeUnregisteredInteractions[i].erase(it);
                change = true;
            }
            if (it != activeUnregisteredInteractions[i].end())
                it++;
        }
    }

    return change;
}

void coInteractionManager::registerInteraction(coInteraction *interaction)
{

    //fprintf(stderr,"+++++++++  coInteractionManager::registerInteraction\n");

    if (interaction->registered)
        return; // this interaction is already registered

    coInteraction::InteractionType type = interaction->getType();

    // search the list backwards
    // compare if the list element interaction prio is less than the new interaction prio
    // if so, insert new interaction after list element
    // otherwise insert at begin (=bottom of stack)
    for (list<coInteraction *>::iterator it = interactionStack[type].end();
         it != interactionStack[type].begin();)
    {

        --it;
        // if the interaction is higher as the one of the first element in list
        if ((*it)->getPriority() <= interaction->getPriority())
        {

            // is this thelast element, if so, remove its icon and show the new one
            if ((*it) == interactionStack[type].back())
            {
                (*it)->removeIcon();
                interaction->addIcon();
            }
            ++it;

            interaction->registered = true;
            // insert before it
            interactionStack[type].insert(it, interaction);

            // if this interaction is still in the list of active unregistered interactions
            // remove it there
            for (list<coInteraction *>::iterator it = activeUnregisteredInteractions[type].begin();
                 it != activeUnregisteredInteractions[type].end();
                 ++it)
            {
                if ((*it) == interaction)
                {
                    activeUnregisteredInteractions[(*it)->getType()].remove(interaction);
                    break;
                }
            }

            return;
        }
    }

    // all interactions on the stack have higher priority than the new one
    // just add the interaction in front of the list (=bottom of the stack)
    interaction->registered = true;
    if (interactionStack[interaction->getType()].empty())
        interaction->addIcon(); // if the list is empty display the icon
    interactionStack[interaction->getType()].push_front(interaction);

    // if this interaction is still in the list of active unregistered interactions
    // remove it there
    for (list<coInteraction *>::iterator it = activeUnregisteredInteractions[type].begin();
         it != activeUnregisteredInteractions[type].end();
         ++it)
    {
        if ((*it) == interaction)
        {
            activeUnregisteredInteractions[(*it)->getType()].remove(interaction);
            break;
        }
    }
}

void coInteractionManager::unregisterInteraction(coInteraction *interaction)
{

    if (!interaction)
        return; // this interaction is not registered

    for (int i = 0; i < coInteraction::NumInteractorTypes; ++i)
    {
        // go through the list of active but unregisted interactions
        // update all interactions in this list
        // if they are not running any more
        // remove it from this list
        for (list<coInteraction *>::iterator it = activeUnregisteredInteractions[i].begin();
             it != activeUnregisteredInteractions[i].end();)
        {
            if ((*it)->getState() == coInteraction::Idle)
            {
                (*it)->resetState();
                it = activeUnregisteredInteractions[(*it)->getType()].erase(it);
            }
            if (it != activeUnregisteredInteractions[i].end())
                it++;
        }
    }

    if (!interaction->registered)
        return; // this interaction is not registered

    // remove the interaction icon
    // dr: sollte fuer active interactions verzoegert werden
    bool removedLast = false;
    if (interactionStack[interaction->getType()].size() && interaction == interactionStack[interaction->getType()].back())
    {
        interaction->removeIcon();
        removedLast = true;
    }
    coButtonInteraction *bi = dynamic_cast<coButtonInteraction *>(interaction);
    // running interactions will not be stopped, but appended to another list
    if ((interaction->getState() != coInteraction::Idle) || (bi && !bi->isIdle())) // allow button interactions to reach an idle state
    {
        activeUnregisteredInteractions[interaction->getType()].push_back(interaction);
    }

    // unregister the interaction and remove it from the stack
    interaction->registered = false;
    interactionStack[interaction->getType()].remove(interaction);

    // give derived classes the changes to clear the state
    if (interaction->getState() == coInteraction::Idle)
    {
        interaction->resetState();
    }
    // if there are several intarctions on the stack, show the last interaction
    if (removedLast && interactionStack[interaction->getType()].size() && interactionStack[interaction->getType()].back())
    {
        interactionStack[interaction->getType()].back()->addIcon();
    }
}
}
