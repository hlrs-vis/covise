/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EVENT_H
#define EVENT_H

#include "EventTypes.h"
#include "EventSender.h"

// forward declaration
class SceneObject;
class Behavior;

class Event
{
public:
    Event();
    virtual ~Event();

    virtual EventTypes::Type getType();

    void setSender(EventSender sender);
    EventSender getSender();

    void setHandled();
    bool wasHandled();

protected:
    EventTypes::Type _type;
    EventSender _sender;
    bool _handled;
};

#endif
