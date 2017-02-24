/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCGEAR_H
#define OSCGEAR_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscGear : public oscObjectBase
{
public:
oscGear()
{
        OSC_ADD_MEMBER(number, 0);
        OSC_ADD_MEMBER(active, 0);
    };
        const char *getScope(){return "/OSCPrivateAction/ActionController/Override";};
    oscDouble number;
    oscBool active;

};

typedef oscObjectVariable<oscGear *> oscGearMember;
typedef oscObjectVariableArray<oscGear *> oscGearArrayMember;


}

#endif //OSCGEAR_H
