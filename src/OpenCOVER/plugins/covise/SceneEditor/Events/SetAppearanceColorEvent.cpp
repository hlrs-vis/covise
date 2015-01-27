/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SetAppearanceColorEvent.h"

SetAppearanceColorEvent::SetAppearanceColorEvent()
{
    _type = EventTypes::SET_APPEARANCE_COLOR_EVENT;
    _scope = "";
    _color = osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

SetAppearanceColorEvent::~SetAppearanceColorEvent()
{
}

void SetAppearanceColorEvent::setScope(std::string scope)
{
    _scope = scope;
}

std::string SetAppearanceColorEvent::getScope()
{
    return _scope;
}

void SetAppearanceColorEvent::setColor(osg::Vec4 color)
{
    _color = color;
}

osg::Vec4 SetAppearanceColorEvent::getColor()
{
    return _color;
}
