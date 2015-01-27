/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SwitchVariantEvent.h"

SwitchVariantEvent::SwitchVariantEvent()
{
    _type = EventTypes::SWITCH_VARIANT_EVENT;
    _group = "";
    _variant = "";
}

SwitchVariantEvent::~SwitchVariantEvent()
{
}

void SwitchVariantEvent::setGroup(std::string group)
{
    _group = group;
}

std::string SwitchVariantEvent::getGroup()
{
    return _group;
}

void SwitchVariantEvent::setVariant(std::string variant)
{
    _variant = variant;
}

std::string SwitchVariantEvent::getVariant()
{
    return _variant;
}
