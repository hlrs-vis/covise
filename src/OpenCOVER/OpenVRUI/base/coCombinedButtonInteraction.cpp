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

    updateState(curbutton);
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
