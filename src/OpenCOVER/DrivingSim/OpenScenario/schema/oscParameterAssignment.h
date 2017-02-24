/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPARAMETERASSIGNMENT_H
#define OSCPARAMETERASSIGNMENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSetParameter.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscParameterAssignment : public oscObjectBase
{
public:
oscParameterAssignment()
{
        OSC_OBJECT_ADD_MEMBER(Parameter, "oscSetParameter", 0);
    };
        const char *getScope(){return "";};
    oscSetParameterMember Parameter;

};

typedef oscObjectVariable<oscParameterAssignment *> oscParameterAssignmentMember;
typedef oscObjectVariableArray<oscParameterAssignment *> oscParameterAssignmentArrayMember;


}

#endif //OSCPARAMETERASSIGNMENT_H
