/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SWITCH_VARIANT_EVENT_H
#define SWITCH_VARIANT_EVENT_H

#include "Event.h"

#include <string>

class SwitchVariantEvent : public Event
{
public:
    SwitchVariantEvent();
    virtual ~SwitchVariantEvent();

    void setGroup(std::string group);
    std::string getGroup();

    void setVariant(std::string variant);
    std::string getVariant();

private:
    std::string _group;
    std::string _variant;
};

#endif
