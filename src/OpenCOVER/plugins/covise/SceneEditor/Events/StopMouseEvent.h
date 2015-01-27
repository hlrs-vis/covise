/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STOP_MOUSE_EVENT_H
#define STOP_MOUSE_EVENT_H

#include "MouseEvent.h"

class StopMouseEvent : public MouseEvent
{
public:
    StopMouseEvent();
    virtual ~StopMouseEvent();
};

#endif
