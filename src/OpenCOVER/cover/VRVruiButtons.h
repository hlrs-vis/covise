/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_VRUI_BUTTONS_H
#define VR_VRUI_BUTTONS_H

/*! \file
 \brief  OpenVRUI interface to OpenCOVER button state

 \author Andreas Kopecki <kopecki@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date  2004
 */

#include <stdlib.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
namespace opencover
{
class coPointerButton;

class COVEREXPORT VRVruiButtons : public vrui::vruiButtons
{
public:
    enum ButtonsType
    {
        Pointer,
        Mouse,
        Relative,
    };

    VRVruiButtons(ButtonsType type=Pointer);
    virtual ~VRVruiButtons();

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
#endif
