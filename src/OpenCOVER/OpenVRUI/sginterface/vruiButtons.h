/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_BUTTONS_H
#define VRUI_BUTTONS_H

#include <util/coTypes.h>
#include "../coInteraction.h"

namespace vrui
{

class OPENVRUIEXPORT vruiButtons
{
public:
    vruiButtons()
    {
    }
    virtual ~vruiButtons();

    enum Button
    {
        NO_BUTTON = 0x0000,
        ACTION_BUTTON = 1<<coInteraction::ButtonAction,
        DRIVE_BUTTON = 1<<coInteraction::ButtonDrive,
        XFORM_BUTTON = 1<<coInteraction::ButtonXform,
        FORWARD_BUTTON = 1<<coInteraction::ButtonForward,
        BACKWARD_BUTTON = 1<<coInteraction::ButtonBack,
        TOGGLE_DOCUMENTS = 1<coInteraction::ButtonToggleDocuments,
        DRAG_BUTTON = 1<<coInteraction::ButtonDrag,
        ZOOM_BUTTON = 1<<coInteraction::ButtonZoom,
        MENU_BUTTON = 1<<coInteraction::ButtonMenu,
        QUIT_BUTTON = 1<<coInteraction::ButtonQuit,
        INTER_NEXT = 1<<coInteraction::ButtonNextInter,
        INTER_PREV = 1<<coInteraction::ButtonPrevInter,
        PERSON_NEXT = 1<<coInteraction::ButtonNextPerson,
        PERSON_PREV = 1<<coInteraction::ButtonPrevPerson,
        WHEEL_UP = 0x01000000,
        WHEEL_DOWN = 0x02000000,
        WHEEL_LEFT = 0x04000000,
        WHEEL_RIGHT = 0x08000000,
        WHEEL = WHEEL_UP | WHEEL_DOWN | WHEEL_LEFT | WHEEL_RIGHT,
        JOYSTICK_RIGHT = 0x10000000,
        JOYSTICK_DOWN = 0x20000000,
        JOYSTICK_LEFT = 0x40000000,
        JOYSTICK_UP = 0x80000000,
        ALL_BUTTONS = ACTION_BUTTON | DRIVE_BUTTON | XFORM_BUTTON | FORWARD_BUTTON | BACKWARD_BUTTON
    };
    static_assert(coInteraction::ButtonPrevPerson == coInteraction::LastButton, "add missing buttons in Buttons enum");

    virtual unsigned int wasPressed(unsigned int buttonMask=ALL_BUTTONS) const = 0;
    virtual unsigned int wasReleased(unsigned int buttonMask=ALL_BUTTONS) const = 0;

    virtual unsigned int getStatus() const = 0;
    virtual unsigned int getOldStatus() const = 0;

    virtual int getWheelCount(size_t idx=0) const = 0;
};
}
#endif
