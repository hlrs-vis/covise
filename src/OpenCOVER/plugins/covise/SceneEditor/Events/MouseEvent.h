/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MOUSE_EVENT_H
#define MOUSE_EVENT_H

#include "Event.h"

class MouseEvent : public Event
{
public:
    enum ButtonType
    {
        TYPE_BUTTON_A,
        TYPE_BUTTON_B,
        TYPE_BUTTON_C
    };

    MouseEvent();
    virtual ~MouseEvent();

    virtual void setMouseButton(ButtonType t);
    virtual ButtonType getMouseButton();

protected:
    ButtonType _mouseButton;
};

#endif
