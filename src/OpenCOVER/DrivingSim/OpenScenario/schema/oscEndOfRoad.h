/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENDOFROAD_H
#define OSCENDOFROAD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEndOfRoad : public oscObjectBase
{
public:
oscEndOfRoad()
{
        OSC_ADD_MEMBER(duration);
    };
    oscDouble duration;

};

typedef oscObjectVariable<oscEndOfRoad *> oscEndOfRoadMember;
typedef oscObjectVariableArray<oscEndOfRoad *> oscEndOfRoadArrayMember;


}

#endif //OSCENDOFROAD_H
