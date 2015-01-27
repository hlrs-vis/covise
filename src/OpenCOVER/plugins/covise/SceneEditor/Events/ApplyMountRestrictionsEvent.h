/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef APPLY_MOUNT_RESTRICTIONS_EVENT_H
#define APPLY_MOUNT_RESTRICTIONS_EVENT_H

#include "Event.h"

#include <osg/Matrix>

class ApplyMountRestrictionsEvent : public Event
{
public:
    ApplyMountRestrictionsEvent();
    virtual ~ApplyMountRestrictionsEvent();

private:
};

#endif
