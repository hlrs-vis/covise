/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLONGITUDINALACTION_H
#define OSCLONGITUDINALACTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSpeed.h"
#include "oscDistance.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLongitudinalAction : public oscObjectBase
{
public:
oscLongitudinalAction()
{
        OSC_OBJECT_ADD_MEMBER(Speed, "oscSpeed", 1);
        OSC_OBJECT_ADD_MEMBER(Distance, "oscDistance", 1);
    };
    oscSpeedMember Speed;
    oscDistanceMember Distance;

};

typedef oscObjectVariable<oscLongitudinalAction *> oscLongitudinalActionMember;
typedef oscObjectVariableArray<oscLongitudinalAction *> oscLongitudinalActionArrayMember;


}

#endif //OSCLONGITUDINALACTION_H
