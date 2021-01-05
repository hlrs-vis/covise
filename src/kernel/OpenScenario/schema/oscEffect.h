/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEFFECT_H
#define OSCEFFECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEffect : public oscObjectBase
{
public:
oscEffect()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(intensity, 0);
    };
        const char *getScope(){return "/OSCEnvironment/RoadCondition";};
    oscString name;
    oscDouble intensity;

};

typedef oscObjectVariable<oscEffect *> oscEffectMember;
typedef oscObjectVariableArray<oscEffect *> oscEffectArrayMember;


}

#endif //OSCEFFECT_H
