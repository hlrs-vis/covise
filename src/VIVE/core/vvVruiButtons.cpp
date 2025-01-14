/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvVruiButtons.h"
#include <cover/vvPluginSupport.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/util/vruiLog.h>

using namespace vive;
using namespace covise;

vvVriuiButtons::vvVriuiButtons(ButtonsType type)
: m_type(type)
{
}

vvVriuiButtons::~vvVriuiButtons()
{
}

coPointerButton *vvVriuiButtons::button() const
{
    switch (m_type)
    {
    case Pointer:
        return vv->getPointerButton();
        break;
    case Mouse:
        return vv->getMouseButton();
        break;
    case Relative:
        return vv->getRelativeButton();
        break;
    }

    return vv->getPointerButton();
}

unsigned int vvVriuiButtons::wasPressed(unsigned int buttons) const
{
    if (!button())
        return 0;

    return button()->wasPressed(buttons);
}

unsigned int vvVriuiButtons::wasReleased(unsigned int buttons) const
{
    if (!button())
        return 0;

    return button()->wasReleased(buttons);
}

unsigned int vvVriuiButtons::getStatus() const
{
    if (!button())
        return 0;

    return button()->getState();
}

unsigned int vvVriuiButtons::getOldStatus() const
{
    if (!button())
        return 0;

    return button()->oldState();
}

int vvVriuiButtons::getWheelCount(size_t idx) const
{
    if (!button())
        return 0;

    return button()->getWheel(idx);
}
