/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRULE_H
#define OSCRULE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscAdd.h"
#include "schema/oscMultiply.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRule : public oscObjectBase
{
public:
    oscRule()
    {
        OSC_OBJECT_ADD_MEMBER(Add, "oscAdd");
        OSC_OBJECT_ADD_MEMBER(Multiply, "oscMultiply");
    };
    oscAddMember Add;
    oscMultiplyMember Multiply;

};

typedef oscObjectVariable<oscRule *> oscRuleMember;


}

#endif //OSCRULE_H
