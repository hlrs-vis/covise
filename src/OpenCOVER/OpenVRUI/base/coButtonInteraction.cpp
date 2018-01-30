/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coButtonInteraction.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

coButtonInteraction::coButtonInteraction(InteractionType type, const string &name, InteractionPriority priority)
    : coInteraction(type, name, priority)
{
    //fprintf(stderr,"coButtonInteraction::coButtonInteraction \n");
    wheelCount = 0;
    button = NULL;
    if (type == AllButtons)
    {
        buttonmask = ~0;
    }
    else if (type == Wheel)
    {
        buttonmask = 0;
    }
    else
    {
        buttonmask = 1<<type;
    }
}

coButtonInteraction::~coButtonInteraction()
{
    //fprintf(stderr,"coButtonInteraction::~coButtonInteraction\n");
    coInteractionManager::the()->unregisterInteraction(this);
}

void coButtonInteraction::update()
{
    updateState(button);
}

bool coButtonInteraction::conditionMet() const
{
    if (type == Wheel)
    {
        if (button->getWheelCount() != 0)
            return true;
    }
    else
    {
        return (buttonmask & button->getStatus());
    }

    return false;
}

bool coButtonInteraction::conditionBecameMet() const
{
    if (type == Wheel)
    {
        if (button->getWheelCount() != 0)
            return true;
    }
    else
    {
        return button->wasPressed(buttonmask);
    }

    return false;
}

void coButtonInteraction::updateState(vruiButtons *button)
{
    runningState = StateNotRunning;

    if (state == Idle)
    {
        //fprintf(stderr,"coButtonInteraction::update state == Idle\n");
        // here whe do the mapping between Interaction types and the button bitmask
        if (conditionBecameMet())
        {
            if (activate())
            {
                //fprintf(stderr,"coButtonInteraction::update 1 StateStarted %s \n", name.c_str());
                //fprintf(stderr,"coButtonInteraction (state == Idle) %d (type == ButtonA || type == AllButtons) %d buttonStatus == vruiButtons::ACTION_BUTTON %d state != Paused %d\n", (state == Idle), (type == ButtonA || type == AllButtons), (buttonStatus == vruiButtons::ACTION_BUTTON), state != Paused);
                runningState = StateStarted;

                if (button && type == Wheel)
                    wheelCount = button->getWheelCount();

                startInteraction();
            }
        }
    }
    else if (state == Paused)
    {
        runningState = StateStopped;
        if (!conditionMet())
        {
            stopInteraction();
            state = Stopped;
        }
    }
    else if (state == Active)
    {
        //fprintf(stderr,"coButtonInteraction::update state == Active || state == Paused\n");
         if (button && type == Wheel)
                wheelCount = button->getWheelCount();

        if (conditionMet())
        {
            //fprintf(stderr,"coButtonInteraction::update 6 StateRunning %s \n", name.c_str());
            //fprintf(stderr,"coButtonInteraction (state == Idle) %d button->wasPressed() %d type == ButtonA || type == AllButtons %d buttonStatus == vruiButtons::ACTION_BUTTON %d activate() %d\n", (state == Idle), button->wasPressed(), (type == ButtonA || type == AllButtons), (buttonStatus == vruiButtons::ACTION_BUTTON), activate() );
           
            runningState = StateRunning;
            doInteraction();
        }
        else
        {
            if (type == Wheel)
            {
                //fprintf(stderr,"coButtonInteraction::update 7 StateStopped\n");
                runningState = StateStopped;
                stopInteraction();
                state = Idle;
            }
            else
            {
                //fprintf(stderr,"coButtonInteraction::update 7 StateStopped\n");
                runningState = StateStopped;
                stopInteraction();
                state = Stopped;
            }
        }
    }
    else if (state == Stopped) // das soll einen Frame verzoegern
    {
        //fprintf(stderr,"coButtonInteraction::update state == Stopped\n");
        state = Idle;
    }
}

void coButtonInteraction::cancelInteraction()
{
    //fprintf(stderr,"coButtonInteraction::cancelInteraction \n");
    if (state == Active || state == Paused)
    {
        runningState = StateNotRunning;
        stopInteraction();
        state = Idle;
        wheelCount = 0;
    }
}

void coButtonInteraction::resetState()
{
    //fprintf(stderr,"coButtonInteraction::resetState\n");
    if (runningState != StateStopped)
        runningState = StateNotRunning;
}

void coButtonInteraction::startInteraction()
{
}

void coButtonInteraction::stopInteraction()
{
}

void coButtonInteraction::doInteraction()
{
}
}
