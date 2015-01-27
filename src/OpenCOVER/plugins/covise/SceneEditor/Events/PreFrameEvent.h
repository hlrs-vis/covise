/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PRE_FRAME_EVENT_H
#define PRE_FRAME_EVENT_H

#include "Event.h"

class PreFrameEvent : public Event
{
public:
    PreFrameEvent();
    virtual ~PreFrameEvent();
};

#endif
