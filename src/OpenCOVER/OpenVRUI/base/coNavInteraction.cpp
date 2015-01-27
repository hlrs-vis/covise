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
    runningState = StateNotRunning;
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
        if (button->wasPressed())
        {
            if (type == ButtonA || type == AllButtons)
            {
                if (button->wasPressed() & vruiButtons::ACTION_BUTTON)
                {
                    if (activate())
                    {
                        runningState = StateStarted;
                        startInteraction();
                    }
                }
            }
            if (type == ButtonB || type == AllButtons)
            {
                if (button->wasPressed() & vruiButtons::DRIVE_BUTTON)
                {
                    if (activate())
                    {
                        runningState = StateStarted;
                        startInteraction();
                    }
                }
            }
            if (type == ButtonC || type == AllButtons)
            {
                if (button->wasPressed() & vruiButtons::XFORM_BUTTON)
                {
                    if (activate())
                    {
                        runningState = StateStarted;
                        startInteraction();
                    }
                }
            }
        }
    }
    else if (state == Active || state == Paused || state == ActiveNotify)
    //else if (state == Active)
    {
        if (type == ButtonA || type == AllButtons)
        {
            if (button->getStatus() & vruiButtons::ACTION_BUTTON)
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
        if (type == ButtonB || type == AllButtons)
        {
            if (button->getStatus() & vruiButtons::DRIVE_BUTTON)
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
        if (type == ButtonC || type == AllButtons)
        {
            if (button->getStatus() & vruiButtons::XFORM_BUTTON)
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
