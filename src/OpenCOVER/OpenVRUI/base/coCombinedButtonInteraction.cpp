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
        if (curbutton->wasPressed(1<<type))
        {
            if (activate())
            {
                runningState = StateStarted;
                startInteraction();
            }
        }
        if ((type & Wheel) && (buttonStatus & Wheel))
        {
            wheelCount = curbutton->getWheelCount();
            if (activate())
            {
                runningState = StateStarted;
                startInteraction();
            }
        }
    }
    else if (state == Active)
    {
        if ((1<<type) & buttonStatus)
        {
            if ((buttonStatus & Wheel) && (type & Wheel))
                wheelCount = curbutton->getWheelCount();
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
