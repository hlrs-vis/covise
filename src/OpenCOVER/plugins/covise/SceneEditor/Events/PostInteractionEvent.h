/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POST_INTERACTION_EVENT_H
#define POST_INTERACTION_EVENT_H

#include "Event.h"

// This event is sent after the user interactis with an object (to the object that was modified).
// The mount behavior sends the event to all children (i.e. to all objects that might have been modified as well).
// Mainly used to send the new transformation to the GUI.
class PostInteractionEvent : public Event
{
public:
    PostInteractionEvent();
    virtual ~PostInteractionEvent();
};

#endif
