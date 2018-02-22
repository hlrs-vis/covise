/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vruiRendererInterface.h"
#include "vruiButtons.h"

namespace vrui
{

vruiRendererInterface *vruiRendererInterface::theInterface = 0;

vruiRendererInterface::vruiRendererInterface()
{
    interactionScaleSensitivity = 1.0;
    upVector = coVector(0.0, 0.0, 1.0);
    ray = true;
}

vruiRendererInterface::~vruiRendererInterface()
{
}

vruiRendererInterface *vruiRendererInterface::the()
{
    return theInterface;
}

vruiButtons *vruiRendererInterface::getButtons() const
{
    return buttons;
}

vruiButtons *vruiRendererInterface::getMouseButtons() const
{
    return mouseButtons;
}

vruiButtons *vruiRendererInterface::getRelativeButtons() const
{
    return relativeButtons;
}

/**
 * the sensitivity must be between 0 (low sensitivity) and 1(full sensitivity). 
 */
void vruiRendererInterface::setInteractionScaleSensitivity(float f)
{
    if (f > 1.0)
        f = 1.0f;
    if (f <= 0.0)
        f = 0.00001f;
    interactionScaleSensitivity = f;
}

bool vruiRendererInterface::isJoystickActive()
{
    if (getJoystickManager())
        return getJoystickManager()->getActive();
    else
        return false;
}

void vruiRendererInterface::setJoystickActvie(bool b)
{
    if (getJoystickManager())
        getJoystickManager()->setActive(b);
}
}
