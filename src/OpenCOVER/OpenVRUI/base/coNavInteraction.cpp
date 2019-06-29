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

    if (getState() == Idle)
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
    else if (getState() == Active || getState() == Paused || getState() == ActiveNotify)
    //else if (state == Active)
    {
        if (button->getStatus() & (1<<type))
        {
            if (getState() == Paused)
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
            setState(Idle);
        }
    }
}

void coNavInteraction::cancelInteraction()
{
    if (getState() == Active || getState() == Paused || getState() == ActiveNotify)
    {
        runningState = StateNotRunning;
        stopInteraction();
        setState(Idle);
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
