/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SettingsChangedEvent.h"

SettingsChangedEvent::SettingsChangedEvent()
{
    _type = EventTypes::SETTINGS_CHANGED_EVENT;
}

SettingsChangedEvent::~SettingsChangedEvent()
{
}
