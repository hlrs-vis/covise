/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MouseEvent.h"

MouseEvent::MouseEvent()
{
    _type = EventTypes::NONE;
    _mouseButton = TYPE_BUTTON_A;
}

MouseEvent::~MouseEvent()
{
}

void MouseEvent::setMouseButton(ButtonType t)
{
    _mouseButton = t;
}

MouseEvent::ButtonType MouseEvent::getMouseButton()
{
    return _mouseButton;
}
