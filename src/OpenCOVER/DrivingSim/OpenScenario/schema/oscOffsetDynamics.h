/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOFFSETDYNAMICS_H
#define OSCOFFSETDYNAMICS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSpeedDynamics.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOffsetDynamics : public oscObjectBase
{
public:
    oscOffsetDynamics()
    {
        OSC_ADD_MEMBER_OPTIONAL(maxLateralAcc);
        OSC_ADD_MEMBER_OPTIONAL(duration);
        OSC_ADD_MEMBER(shape);
    };
    oscDouble maxLateralAcc;
    oscDouble duration;
    oscEnum shape;

};

typedef oscObjectVariable<oscOffsetDynamics *> oscOffsetDynamicsMember;


}

#endif //OSCOFFSETDYNAMICS_H
