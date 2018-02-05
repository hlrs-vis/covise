/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coNavInteraction.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

using namespace std;

namespace vrui
{

coNavInteraction::coNavInteraction(InteractionType type, const string &name, InteractionPriority priority)
    : coInteraction(type, name, priority)
{
    oldState = Idle;
    group = GroupNavigation;
}

coNavInteraction::~coNavInteraction()
{
    coInteractionManager::the()->unregisterInteraction(this);
}

void coNavInteraction::update()
{

    vruiButtons *button = vruiRendererInterface::the()->getButtons();

    runningState = StateNotRunning;

    if (state == Idle)
    {
        if (button->wasPressed(1<<type))
        {
            if (activate())
            {
                runningState = StateStarted;
                startInteraction();
            }
        }
    }
    else if (state == Active || state == Paused || state == ActiveNotify)
    //else if (state == Active)
    {
        if (button->getStatus() & (1<<type))
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
        }
        else
        {
            runningState = StateStopped;
            stopInteraction();
            state = Idle;
        }
    }
}

void coNavInteraction::cancelInteraction()
{
    if (state == Active || state == Paused || state == ActiveNotify)
    {
        runningState = StateNotRunning;
        stopInteraction();
        state = Idle;
    }
}

void coNavInteraction::startInteraction()
{
}

void coNavInteraction::stopInteraction()
{
}

void coNavInteraction::doInteraction()
{
}
}
