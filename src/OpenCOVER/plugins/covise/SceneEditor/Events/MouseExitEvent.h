/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MOUSE_EXIT_EVENT_H
#define MOUSE_EXIT_EVENT_H

#include "Event.h"

class MouseExitEvent : public Event
{
public:
    MouseExitEvent();
    virtual ~MouseExitEvent();
};

#endif
