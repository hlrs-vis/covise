/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_BUTTONS_H
#define VRUI_BUTTONS_H

#include <util/coTypes.h>

namespace vrui
{

class OPENVRUIEXPORT vruiButtons
{
public:
    vruiButtons()
    {
    }
    virtual ~vruiButtons()
    {
    }

    enum Button
    {
        NO_BUTTON = 0x0000,
        ACTION_BUTTON = 0x0001,
        DRIVE_BUTTON = 0x0002,
        XFORM_BUTTON = 0x0004,
        USER1_BUTTON = 0x0008,
        USER4_BUTTON = 0x0010,
        TOGGLE_DOCUMENTS = 0x20,
        INTER_PREV = 0x0040,
        INTER_NEXT = 0x0080,
        MENU_BUTTON = 0x0100,
        FORWARD_BUTTON = 0x0200,
        BACKWARD_BUTTON = 0x0400,
        ZOOM_BUTTON = 0x0800,
        QUIT_BUTTON = 0x1000,
        DRAG_BUTTON = 0x2000,
        WHEEL_UP = 0x4000,
        WHEEL_DOWN = 0x8000,
        WHEEL = WHEEL_UP | WHEEL_DOWN,
        PERSON_PREV = 0x10000,
        PERSON_NEXT = 0x20000,
        JOYSTICK_RIGHT = 0x10000000,
        JOYSTICK_DOWN = 0x20000000,
        JOYSTICK_LEFT = 0x40000000,
        JOYSTICK_UP = 0x80000000,
        ALL_BUTTONS = ACTION_BUTTON | DRIVE_BUTTON | XFORM_BUTTON | USER1_BUTTON | USER4_BUTTON
    };

    virtual unsigned int wasPressed(unsigned int buttonMask=ALL_BUTTONS) const = 0;
    virtual unsigned int wasReleased(unsigned int buttonMask=ALL_BUTTONS) const = 0;

    virtual unsigned int getStatus() const = 0;
    virtual unsigned int getOldStatus() const = 0;

    virtual int getWheelCount() const = 0;
};
}
#endif
