/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTIMEOFDAY_H
#define OSCTIMEOFDAY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscTimeHeadway.h"
#include "schema/oscTime.h"
#include "schema/oscDate.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTimeOfDay : public oscObjectBase
{
public:
    oscTimeOfDay()
    {
        OSC_ADD_MEMBER(animation);
        OSC_ADD_MEMBER(rule);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Time, "oscTime");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Date, "oscDate");
    };
    oscBool animation;
    oscEnum rule;
    oscTimeMember Time;
    oscDateMember Date;

};

typedef oscObjectVariable<oscTimeOfDay *> oscTimeOfDayMember;


}

#endif //OSCTIMEOFDAY_H
