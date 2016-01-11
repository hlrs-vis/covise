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
    oldState = Idle;
    runningState = StateNotRunning;
    wheelCount = 0;
    button = NULL;
}

coButtonInteraction::~coButtonInteraction()
{
    //fprintf(stderr,"coButtonInteraction::~coButtonInteraction\n");
    coInteractionManager::the()->unregisterInteraction(this);
}

void coButtonInteraction::update()
{
    if (!button)
        return;

    /*   if(button->wasPressed())
   {
   if(buttonStatustype == ButtonA)
      fprintf(stderr,"coButtonInteraction ButtonA \n");
   if(type == ButtonB)
      fprintf(stderr,"coButtonInteraction ButtonB \n");
   if(type == ButtonC)
      fprintf(stderr,"coButtonInteraction ButtonC \n");
   if(type == ButtonD)
      fprintf(stderr,"coButtonInteraction ButtonD \n");
   if(type == ButtonE)
      fprintf(stderr,"coButtonInteraction ButtonE \n");
   if(type == Wheel)
      fprintf(stderr,"coButtonInteraction Wheel \n");
   if(type == Joystick)
      fprintf(stderr,"coButtonInteraction Joystick \n");
   }*/
    //fprintf(stderr,"coButtonInteraction::update() %s \n", name.c_str());
    //cerr << "run " << isSetRunning << "   this " << this << endl;

    runningState = StateNotRunning;
    unsigned int buttonStatus = button->getStatus();
    //fprintf(stderr,"buttonStatus %d \n", buttonStatus);
    //    if(buttonStatus == vruiButtons::ACTION_BUTTON)
    //       fprintf(stderr,"  vruiButtons::ACTION_BUTTON\n");
    //    if(buttonStatus == vruiButtons::DRAG_BUTTON)
    //       fprintf(stderr,"  vruiButtons::DRAG_BUTTON\n");
    //    if(buttonStatus == vruiButtons::DRIVE_BUTTON)
    //       fprintf(stderr,"  vruiButtons::DRIVE_BUTTON\n");
    //    if(buttonStatus == vruiButtons::XFORM_BUTTON)
    //       fprintf(stderr,"  vruiButtons::XFORM_BUTTON\n");
    //    if(buttonStatus == vruiButtons::WHEEL)
    //       fprintf(stderr,"  vruiButtons::WHEEL\n");
    //    if(buttonStatus == vruiButtons::WHEEL_DOWN)
    //       fprintf(stderr,"  vruiButtons::WHEEL_DOWN\n");
    //    if(buttonStatus == vruiButtons::WHEEL_UP)
    //       fprintf(stderr,"  vruiButtons::WHEEL_UP\n");

    if (state == Idle)
    {
        //fprintf(stderr,"coButtonInteraction::update state == Idle\n");
        // here whe do the mapping between Interaction types and the button bitmask
        if (type == Wheel)
        {
            if (activate())
            {
                if (buttonStatus & vruiButtons::WHEEL)
                {
                    runningState = StateStarted;
                    startInteraction();
                }
                wheelCount = button->getWheelCount();
            }
        }
        else
        {
            if (button->wasPressed(1<<type))
            {
                if (activate())
                {
                    //fprintf(stderr,"coButtonInteraction::update 1 StateStarted %s \n", name.c_str());
                    //fprintf(stderr,"coButtonInteraction (state == Idle) %d (type == ButtonA || type == AllButtons) %d buttonStatus == vruiButtons::ACTION_BUTTON %d state != Paused %d\n", (state == Idle), (type == ButtonA || type == AllButtons), (buttonStatus == vruiButtons::ACTION_BUTTON), state != Paused);
                    runningState = StateStarted;
                    startInteraction();
                }
            }
        }
    }
    else if (state == Paused)
    {
        runningState = StateStopped;
        if (!((1<<type) & buttonStatus))
        {
            stopInteraction();
            state = Stopped;
        }
    }
    else if (state == Active)
    {
        //fprintf(stderr,"coButtonInteraction::update state == Active || state == Paused\n");
         if (type == Wheel)
                wheelCount = button->getWheelCount();

        if ((1<<type) & buttonStatus)
        {
            //fprintf(stderr,"coButtonInteraction::update 6 StateRunning %s \n", name.c_str());
            //fprintf(stderr,"coButtonInteraction (state == Idle) %d button->wasPressed() %d type == ButtonA || type == AllButtons %d buttonStatus == vruiButtons::ACTION_BUTTON %d activate() %d\n", (state == Idle), button->wasPressed(), (type == ButtonA || type == AllButtons), (buttonStatus == vruiButtons::ACTION_BUTTON), activate() );
           
            runningState = StateRunning;
            doInteraction();
        }
        else
        {
            //fprintf(stderr,"coButtonInteraction::update 7 StateStopped\n");
            runningState = StateStopped;
            stopInteraction();
            state = Stopped;
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
