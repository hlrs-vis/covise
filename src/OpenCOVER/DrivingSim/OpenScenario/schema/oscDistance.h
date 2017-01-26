/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDISTANCE_H
#define OSCDISTANCE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDistanceDynamics.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDistance : public oscObjectBase
{
public:
oscDistance()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER(distance, 0);
        OSC_ADD_MEMBER(timeGap, 0);
        OSC_ADD_MEMBER(freespace, 0);
        OSC_OBJECT_ADD_MEMBER(DistanceDynamics, "oscDistanceDynamics", 0);
    };
    oscString object;
    oscDouble distance;
    oscDouble timeGap;
    oscBool freespace;
    oscDistanceDynamicsMember DistanceDynamics;

};

typedef oscObjectVariable<oscDistance *> oscDistanceMember;
typedef oscObjectVariableArray<oscDistance *> oscDistanceArrayMember;


}

#endif //OSCDISTANCE_H
