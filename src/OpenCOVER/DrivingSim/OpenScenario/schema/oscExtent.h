/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEXTENT_H
#define OSCEXTENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscExtent : public oscObjectBase
{
public:
oscExtent()
{
        OSC_ADD_MEMBER_OPTIONAL(time, 0);
        OSC_ADD_MEMBER_OPTIONAL(distance, 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral/LaneChange/LaneChangeDynamics";};
    oscDouble time;
    oscDouble distance;

};

typedef oscObjectVariable<oscExtent *> oscExtentMember;
typedef oscObjectVariableArray<oscExtent *> oscExtentArrayMember;


}

#endif //OSCEXTENT_H
