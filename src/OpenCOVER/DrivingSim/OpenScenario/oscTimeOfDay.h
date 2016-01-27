/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TIME_OF_DAY_H
#define OSC_TIME_OF_DAY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscTime.h"
#include "oscDate.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTimeOfDay: public oscObjectBase
{
public:
    oscTimeOfDay()
    {
        OSC_OBJECT_ADD_MEMBER(time, "oscTime");
        OSC_OBJECT_ADD_MEMBER(date, "oscDate");
        OSC_ADD_MEMBER(animation);
    };

    oscTimeMember time;
    oscDateMember date;
    oscBool animation;
};

typedef oscObjectVariable<oscTimeOfDay *> oscTimeOfDayMember;

}

#endif //OSC_TIME_OF_DAY_H
