/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDISTANCEDYNAMICS_H
#define OSCDISTANCEDYNAMICS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscEmpty.h"
#include "oscLimited.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDistanceDynamics : public oscObjectBase
{
public:
oscDistanceDynamics()
{
        OSC_OBJECT_ADD_MEMBER(None, "oscEmpty", 1);
        OSC_OBJECT_ADD_MEMBER(Limited, "oscLimited", 1);
    };
    oscEmptyMember None;
    oscLimitedMember Limited;

};

typedef oscObjectVariable<oscDistanceDynamics *> oscDistanceDynamicsMember;
typedef oscObjectVariableArray<oscDistanceDynamics *> oscDistanceDynamicsArrayMember;


}

#endif //OSCDISTANCEDYNAMICS_H
