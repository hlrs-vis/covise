/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// OSG:
#include <iostream>
#include <osgDB/ReadFile>

// Local:
#include "CheckBox.h"
#include "InputDevice.h"

using namespace osg;
using namespace cui;
using namespace std;

CheckBox::CheckBox(Interaction *interaction)
    : Button(interaction)
{
    _isChecked = false;
    _imageChecked = osgDB::readImageFile(_resourcePath + "checkbox-checked.tif");
    _imageUnchecked = osgDB::readImageFile(_resourcePath + "checkbox-unchecked.tif");
    if (!_imageChecked || !_imageUnchecked)
        cerr << "Warning: checkbox icon missing" << endl;
    else
        updateIcon();
}

/** Set the check box state. If triggerEvent is true, 
  the event callback will be called additionally.
*/
void CheckBox::setChecked(bool checked, bool triggerEvent)
{
    _isChecked = checked;
    updateIcon();
    if (triggerEvent)
    {
        std::list<CardListener *>::iterator iter;
        for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
        {
            (*iter)->cardButtonEvent(this, 0, 1);
        }
    }
}

bool CheckBox::isChecked()
{
    return _isChecked;
}

void CheckBox::updateIcon()
{
    if (_isChecked)
    {
        if (_imageChecked)
        {
            Image *image = new Image(*_imageChecked, CopyOp::SHALLOW_COPY);
            _icon->setImage(image);
        }
    }
    else
    {
        if (_imageUnchecked)
        {
            Image *image = new Image(*_imageUnchecked, CopyOp::SHALLOW_COPY);
            _icon->setImage(image);
        }
    }
}

void CheckBox::buttonEvent(InputDevice *evt, int button)
{
    std::list<CardListener *>::iterator iter;
    if (button == 0)
    {
        if (evt->getButtonState(button) & 1)
        {
            for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
            {
                (*iter)->cardButtonEvent(this, 0, 1); // do event code
            }
        }
        else if (!(evt->getButtonState(button) & 7))
        {
            _isChecked = !_isChecked; // change appropriately
            updateIcon();
            for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
            {
                (*iter)->cardButtonEvent(this, 0, 0); // check for external handling
            }
        }
    }
}

/** Set custom images for checked and unchecked states to effectively
  use checkbox as toggle button.
*/
void CheckBox::setImages(std::string checked, std::string unchecked)
{
    _imageChecked = osgDB::readImageFile(checked);
    _imageUnchecked = osgDB::readImageFile(unchecked);
    if (!_imageChecked || !_imageUnchecked)
        cerr << "Warning: custom checkbox icon missing" << endl;
    else
        updateIcon();
}
