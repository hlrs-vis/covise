/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCLOTHOID_H
#define OSCCLOTHOID_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscClothoid : public oscObjectBase
{
public:
oscClothoid()
{
        OSC_ADD_MEMBER(curvature, 0);
        OSC_ADD_MEMBER(curvatureDot, 0);
        OSC_ADD_MEMBER(length, 0);
    };
    oscDouble curvature;
    oscDouble curvatureDot;
    oscDouble length;

};

typedef oscObjectVariable<oscClothoid *> oscClothoidMember;
typedef oscObjectVariableArray<oscClothoid *> oscClothoidArrayMember;


}

#endif //OSCCLOTHOID_H
