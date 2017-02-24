/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLONGITUDINAL_H
#define OSCLONGITUDINAL_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSpeed.h"
#include "oscDistanceAction.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLongitudinal : public oscObjectBase
{
public:
oscLongitudinal()
{
        OSC_OBJECT_ADD_MEMBER(Speed, "oscSpeed", 1);
        OSC_OBJECT_ADD_MEMBER(Distance, "oscDistanceAction", 1);
    };
        const char *getScope(){return "/OSCPrivateAction";};
    oscSpeedMember Speed;
    oscDistanceActionMember Distance;

};

typedef oscObjectVariable<oscLongitudinal *> oscLongitudinalMember;
typedef oscObjectVariableArray<oscLongitudinal *> oscLongitudinalArrayMember;


}

#endif //OSCLONGITUDINAL_H
