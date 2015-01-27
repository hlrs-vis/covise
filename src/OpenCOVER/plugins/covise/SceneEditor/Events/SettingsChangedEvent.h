/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SETTINGS_CHANGED_EVENT_H
#define SETTINGS_CHANGED_EVENT_H

#include "Event.h"

class SettingsChangedEvent : public Event
{
public:
    SettingsChangedEvent();
    virtual ~SettingsChangedEvent();
};

#endif
