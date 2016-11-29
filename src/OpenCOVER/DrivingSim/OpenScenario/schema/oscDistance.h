/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDISTANCE_H
#define OSCDISTANCE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDistanceDynamics.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDistance : public oscObjectBase
{
public:
    oscDistance()
    {
        OSC_ADD_MEMBER(object);
        OSC_ADD_MEMBER(distance);
        OSC_ADD_MEMBER(timeGap);
        OSC_ADD_MEMBER(freespace);
        OSC_OBJECT_ADD_MEMBER(DistanceDynamics, "oscDistanceDynamics");
    };
    oscString object;
    oscDouble distance;
    oscDouble timeGap;
    oscBool freespace;
    oscDistanceDynamicsMember DistanceDynamics;

};

typedef oscObjectVariable<oscDistance *> oscDistanceMember;


}

#endif //OSCDISTANCE_H
