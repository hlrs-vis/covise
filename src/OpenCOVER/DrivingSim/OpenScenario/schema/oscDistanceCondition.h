/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDISTANCECONDITION_H
#define OSCDISTANCECONDITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscTimeHeadway.h"
#include "schema/oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDistanceCondition : public oscObjectBase
{
public:
    oscDistanceCondition()
    {
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(freespace);
        OSC_ADD_MEMBER(alongRoute);
        OSC_ADD_MEMBER(rule);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition");
    };
    oscDouble value;
    oscBool freespace;
    oscBool alongRoute;
    oscEnum rule;
    oscPositionMember Position;

};

typedef oscObjectVariable<oscDistanceCondition *> oscDistanceConditionMember;


}

#endif //OSCDISTANCECONDITION_H
