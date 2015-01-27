/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osgDB/ReadFile>

#include "RadioButton.h"
#include "InputDevice.h"

using namespace cui;

RadioButton::RadioButton(Interaction *interaction)
    : CheckBox(interaction)
{
    _imageChecked = osgDB::readImageFile(_resourcePath + "radiobutton.tif");
    _imageUnchecked = osgDB::readImageFile(_resourcePath + "checkbox-unchecked.tif");
    _isChecked = true;
    updateIcon();
}

void RadioButton::buttonEvent(InputDevice *dev, int button)
{
    std::list<CardListener *>::iterator iter;

    Card::buttonEvent(dev, button);

    if ((button == 0) && (dev->getButtonState(button) == 0))
    {
        _isChecked = true;
        updateIcon();
    }
}
