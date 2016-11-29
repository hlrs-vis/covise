/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDISTANCEDYNAMICS_H
#define OSCDISTANCEDYNAMICS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscNone.h"
#include "schema/oscLimited.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDistanceDynamics : public oscObjectBase
{
public:
    oscDistanceDynamics()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(None, "oscNone");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Limited, "oscLimited");
    };
    oscNoneMember None;
    oscLimitedMember Limited;

};

typedef oscObjectVariable<oscDistanceDynamics *> oscDistanceDynamicsMember;


}

#endif //OSCDISTANCEDYNAMICS_H
