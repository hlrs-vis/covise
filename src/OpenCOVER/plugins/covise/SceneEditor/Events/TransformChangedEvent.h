/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TRANSFORM_CHANGED_EVENT_H
#define TRANSFORM_CHANGED_EVENT_H

#include "Event.h"

class TransformChangedEvent : public Event
{
public:
    TransformChangedEvent();
    virtual ~TransformChangedEvent();
};

#endif
