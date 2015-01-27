/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TouchInteraction.h"

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

TouchInteraction::TouchInteraction(InteractionType type, const std::string &name, InteractionPriority priority)
    : coInteraction(type, name, priority)
    , activationRequested(false)
    , deactivationRequested(false)
{
    oldState = Idle;
    runningState = StateNotRunning;
}

TouchInteraction::~TouchInteraction()
{
    coInteractionManager::the()->unregisterInteraction(this);
}

void TouchInteraction::requestActivation()
{
    activationRequested = true;
}

void TouchInteraction::requestDeactivation()
{
    deactivationRequested = true;
}

void TouchInteraction::update()
{
    runningState = StateNotRunning;

    if (state == Idle)
    {
        if (activationRequested)
        {
            //      std::cout << "activate request" << std::endl;

            activationRequested = false;
            if (activate())
            {
                //            std::cout << "TouchI:update() activate success" << state << std::endl;
                runningState = StateStarted;
                startInteraction();
            }
        }
    }
    else if (state == Active || state == Paused || state == ActiveNotify)
    {
        if (state == Paused)
        {
            runningState = StateStopped;
        }
        else
        {
            runningState = StateRunning;
            doInteraction();
        }

        if (deactivationRequested)
        {
            deactivationRequested = false;
            runningState = StateStopped;
            stopInteraction();
            state = Idle;
        }
    }
}

void TouchInteraction::cancelInteraction()
{
    if (state == Active || state == Paused || state == ActiveNotify)
    {
        runningState = StateNotRunning;
        stopInteraction();
        state = Idle;
    }
}

void TouchInteraction::startInteraction()
{
}

void TouchInteraction::stopInteraction()
{
}

void TouchInteraction::doInteraction()
{
}
