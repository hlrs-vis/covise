/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SELECT_EVENT_H
#define SELECT_EVENT_H

#include "Event.h"

class SelectEvent : public Event
{
public:
    SelectEvent();
    virtual ~SelectEvent();
};

#endif
