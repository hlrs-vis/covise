/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCREACHPOSITION_H
#define OSCREACHPOSITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscReachPosition : public oscObjectBase
{
public:
    oscReachPosition()
    {
        OSC_ADD_MEMBER(tolerance);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition");
    };
    oscDouble tolerance;
    oscPositionMember Position;

};

typedef oscObjectVariable<oscReachPosition *> oscReachPositionMember;


}

#endif //OSCREACHPOSITION_H
