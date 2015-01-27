/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Event.h"

Event::Event()
{
    _type = EventTypes::NONE;
    _sender = EventSender();
    _handled = false;
}

Event::~Event()
{
}

EventTypes::Type Event::getType()
{
    return _type;
}

void Event::setSender(EventSender sender)
{
    _sender = sender;
}

EventSender Event::getSender()
{
    return _sender;
}

void Event::setHandled()
{
    _handled = true;
}

bool Event::wasHandled()
{
    return _handled;
}
