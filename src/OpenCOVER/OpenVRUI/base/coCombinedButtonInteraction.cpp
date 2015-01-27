/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

namespace vrui
{

coCombinedButtonInteraction::coCombinedButtonInteraction(InteractionType type,
                                                         const std::string &name, InteractionPriority priority)
    : coButtonInteraction(type, name, priority)
    , mousebutton(NULL)
    , mouse(false)
{
}

coCombinedButtonInteraction::~coCombinedButtonInteraction()
{
}

void coCombinedButtonInteraction::setHitByMouse(bool isMouse)
{
    mouse = isMouse;
}

bool coCombinedButtonInteraction::isMouse() const
{
    return mouse;
}

void coCombinedButtonInteraction::update()
{
    if (!mousebutton)
        mousebutton = vruiRendererInterface::the()->getMouseButtons();

    if (!mousebutton)
        return;

    if (!button)
        button = vruiRendererInterface::the()->getButtons();

    if (!button)
        button = mousebutton;

    vruiButtons *curbutton = mouse ? mousebutton : button;
    runningState = StateNotRunning;
    unsigned int buttonStatus = curbutton->getStatus();

    if (state == Idle)
    {
        if (curbutton->wasPressed())
        {
            if (type == ButtonA || type == AllButtons)
            {
                if (curbutton->wasPressed() & vruiButtons::ACTION_BUTTON)
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
                if (curbutton->wasPressed() & vruiButtons::DRIVE_BUTTON)
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
                if (curbutton->wasPressed() & vruiButtons::XFORM_BUTTON)
                {
                    if (activate())
                    {
                        runningState = StateStarted;
                        startInteraction();
                    }
                }
            }
        }
        if (type == Wheel || type == AllButtons)
        {
            if (activate())
            {
                if (buttonStatus & vruiButtons::WHEEL_UP || buttonStatus & vruiButtons::WHEEL_DOWN)
                {
                    runningState = StateStarted;
                    startInteraction();
                }

                wheelCount = curbutton->getWheelCount();
            }
        }
    }
    else if (state == Active)
    {
        if (type == ButtonA || type == AllButtons)
        {
            if (buttonStatus & vruiButtons::ACTION_BUTTON)
            {
                runningState = StateRunning;
                doInteraction();
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
            if (buttonStatus & vruiButtons::DRIVE_BUTTON)
            {
                runningState = StateRunning;
                doInteraction();
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
            if (buttonStatus & vruiButtons::XFORM_BUTTON)
            {
                runningState = StateRunning;
                doInteraction();
            }
            else
            {
                runningState = StateStopped;
                stopInteraction();
                state = Idle;
            }
        }
        if (type == Wheel || type == AllButtons)
        {
            if (buttonStatus & (vruiButtons::WHEEL_UP | vruiButtons::WHEEL_DOWN))
            {
                runningState = StateRunning;
                //VRUILOG("doInteraction " << wheelDirection)

                wheelCount = curbutton->getWheelCount();

                doInteraction();
            }
            else
            {
                runningState = StateStopped;
                //VRUILOG("stopInteraction")
                stopInteraction();
                state = Idle;
            }
        }
    }
    else if (state == Stopped) // das soll einen Frame verzoegern
    {
        //fprintf(stderr,"coButtonInteraction::update state == Stopped\n");
        state = Idle;
    }
}

bool coCombinedButtonInteraction::is2D() const
{
    if (mouse)
        return true;

    if (vruiRendererInterface::the()->is2DInputDevice())
        return true;

    return false;
}

vruiMatrix *coCombinedButtonInteraction::getHandMatrix() const
{
    if (mouse)
        return vruiRendererInterface::the()->getMouseMatrix();

    return vruiRendererInterface::the()->getHandMatrix();
}
}
