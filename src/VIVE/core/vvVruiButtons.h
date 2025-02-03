/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <stdlib.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
namespace vive
{
class coPointerButton;

class VVCORE_EXPORT vvVruiButtons : public vrui::vruiButtons
{
public:
    enum ButtonsType
    {
        Pointer,
        Mouse,
        Relative,
    };

    vvVruiButtons(ButtonsType type=Pointer);
    virtual ~vvVruiButtons();

    virtual unsigned int wasPressed(unsigned int buttons) const;
    virtual unsigned int wasReleased(unsigned int buttons) const;

    virtual unsigned int getStatus() const;
    virtual unsigned int getOldStatus() const;

    virtual int getWheelCount(size_t idx=0) const;

private:
    ButtonsType m_type = Pointer;
    coPointerButton *button() const;
};
}
