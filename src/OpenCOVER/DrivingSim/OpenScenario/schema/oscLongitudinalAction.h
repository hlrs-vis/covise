/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLONGITUDINALACTION_H
#define OSCLONGITUDINALACTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSpeed.h"
#include "schema/oscDistance.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLongitudinalAction : public oscObjectBase
{
public:
    oscLongitudinalAction()
    {
        OSC_OBJECT_ADD_MEMBER(Speed, "oscSpeed");
        OSC_OBJECT_ADD_MEMBER(Distance, "oscDistance");
    };
    oscSpeedMember Speed;
    oscDistanceMember Distance;

};

typedef oscObjectVariable<oscLongitudinalAction *> oscLongitudinalActionMember;


}

#endif //OSCLONGITUDINALACTION_H
