/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROAD_H
#define OSCROAD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoad : public oscObjectBase
{
public:
oscRoad()
{
        OSC_ADD_MEMBER(roadId);
        OSC_ADD_MEMBER(s);
        OSC_ADD_MEMBER(t);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation");
    };
    oscString roadId;
    oscDouble s;
    oscDouble t;
    oscOrientationMember Orientation;

};

typedef oscObjectVariable<oscRoad *> oscRoadMember;
typedef oscObjectVariableArray<oscRoad *> oscRoadArrayMember;


}

#endif //OSCROAD_H
