/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDISTANCEACTION_H
#define OSCDISTANCEACTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDynamics.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDistanceAction : public oscObjectBase
{
public:
oscDistanceAction()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER_OPTIONAL(distance, 0);
        OSC_ADD_MEMBER_OPTIONAL(timeGap, 0);
        OSC_ADD_MEMBER(freespace, 0);
        OSC_OBJECT_ADD_MEMBER(Dynamics, "oscDynamics", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Longitudinal";};
    oscString object;
    oscDouble distance;
    oscDouble timeGap;
    oscBool freespace;
    oscDynamicsMember Dynamics;

};

typedef oscObjectVariable<oscDistanceAction *> oscDistanceActionMember;
typedef oscObjectVariableArray<oscDistanceAction *> oscDistanceActionArrayMember;


}

#endif //OSCDISTANCEACTION_H
