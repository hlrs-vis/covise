/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROADCOORD_H
#define OSCROADCOORD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoadCoord : public oscObjectBase
{
public:
oscRoadCoord()
{
        OSC_ADD_MEMBER(pathS);
        OSC_ADD_MEMBER(t);
    };
    oscDouble pathS;
    oscDouble t;

};

typedef oscObjectVariable<oscRoadCoord *> oscRoadCoordMember;
typedef oscObjectVariableArray<oscRoadCoord *> oscRoadCoordArrayMember;


}

#endif //OSCROADCOORD_H
