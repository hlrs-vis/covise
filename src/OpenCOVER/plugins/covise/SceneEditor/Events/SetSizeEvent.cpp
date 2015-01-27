/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SetSizeEvent.h"

SetSizeEvent::SetSizeEvent()
{
    _type = EventTypes::SET_SIZE_EVENT;
    _width = 0;
    _height = 0;
    _length = 0;
}

SetSizeEvent::~SetSizeEvent()
{
}

float SetSizeEvent::getWidth()
{
    return _width;
}

void SetSizeEvent::setWidth(float w)
{
    _width = w;
}

float SetSizeEvent::getHeight()
{
    return _height;
}

void SetSizeEvent::setHeight(float h)
{
    _height = h;
}

float SetSizeEvent::getLength()
{
    return _length;
}

void SetSizeEvent::setLength(float l)
{
    _length = l;
}
