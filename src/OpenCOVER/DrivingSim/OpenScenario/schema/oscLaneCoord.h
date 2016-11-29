/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANECOORD_H
#define OSCLANECOORD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneCoord : public oscObjectBase
{
public:
    oscLaneCoord()
    {
        OSC_ADD_MEMBER(pathS);
        OSC_ADD_MEMBER(laneId);
        OSC_ADD_MEMBER_OPTIONAL(laneOffset);
    };
    oscDouble pathS;
    oscInt laneId;
    oscDouble laneOffset;

};

typedef oscObjectVariable<oscLaneCoord *> oscLaneCoordMember;


}

#endif //OSCLANECOORD_H
