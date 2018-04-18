/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRVruiButtons.h"
#include <cover/coVRPluginSupport.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/util/vruiLog.h>

using namespace opencover;
using namespace covise;

VRVruiButtons::VRVruiButtons(ButtonsType type)
: m_type(type)
{
}

VRVruiButtons::~VRVruiButtons()
{
}

coPointerButton *VRVruiButtons::button() const
{
    switch (m_type)
    {
    case Pointer:
        return cover->getPointerButton();
        break;
    case Mouse:
        return cover->getMouseButton();
        break;
    case Relative:
        return cover->getRelativeButton();
        break;
    }

    return cover->getPointerButton();
}

unsigned int VRVruiButtons::wasPressed(unsigned int buttons) const
{
    if (!button())
        return 0;

    return button()->wasPressed(buttons);
}

unsigned int VRVruiButtons::wasReleased(unsigned int buttons) const
{
    if (!button())
        return 0;

    return button()->wasReleased(buttons);
}

unsigned int VRVruiButtons::getStatus() const
{
    if (!button())
        return 0;

    return button()->getState();
}

unsigned int VRVruiButtons::getOldStatus() const
{
    if (!button())
        return 0;

    return button()->oldState();
}

int VRVruiButtons::getWheelCount(size_t idx) const
{
    if (!button())
        return 0;

    return button()->getWheel(idx);
}
