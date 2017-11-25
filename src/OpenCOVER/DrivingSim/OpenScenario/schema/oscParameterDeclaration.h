/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPARAMETERDECLARATION_H
#define OSCPARAMETERDECLARATION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameter.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscParameterDeclaration : public oscObjectBase
{
public:
oscParameterDeclaration()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Parameter, "oscParameter", 0);
    };
        const char *getScope(){return "";};
    oscParameterArrayMember Parameter;

};

typedef oscObjectVariable<oscParameterDeclaration *> oscParameterDeclarationMember;
typedef oscObjectVariableArray<oscParameterDeclaration *> oscParameterDeclarationArrayMember;


}

#endif //OSCPARAMETERDECLARATION_H
