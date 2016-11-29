/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVEROAD_H
#define OSCRELATIVEROAD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelativeRoad : public oscObjectBase
{
public:
    oscRelativeRoad()
    {
        OSC_ADD_MEMBER(object);
        OSC_ADD_MEMBER(ds);
        OSC_ADD_MEMBER(dt);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation");
    };
    oscString object;
    oscDouble ds;
    oscDouble dt;
    oscOrientationMember Orientation;

};

typedef oscObjectVariable<oscRelativeRoad *> oscRelativeRoadMember;


}

#endif //OSCRELATIVEROAD_H
