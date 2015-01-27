/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FakedMouseEvent.h"

FakedMouseEvent::FakedMouseEvent(int eventType, int x, int y)
{
    this->eventType = eventType;
    this->xPos = x;
    this->yPos = y;
}

FakedMouseEvent::~FakedMouseEvent() {}

int FakedMouseEvent::getEventType() const
{
    return eventType;
}

int FakedMouseEvent::getXPos() const
{
    return xPos;
}

int FakedMouseEvent::getYPos() const
{
    return yPos;
}
