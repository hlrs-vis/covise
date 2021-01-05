/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPEDESTRIANCONTROLLER_H
#define OSCPEDESTRIANCONTROLLER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameterDeclaration.h"
#include "oscPersonDescription.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPedestrianController : public oscObjectBase
{
public:
oscPedestrianController()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ParameterDeclaration, "oscParameterDeclaration", 0);
        OSC_OBJECT_ADD_MEMBER(Description, "oscPersonDescription", 0);
    };
        const char *getScope(){return "";};
    oscString name;
    oscParameterDeclarationMember ParameterDeclaration;
    oscPersonDescriptionMember Description;

};

typedef oscObjectVariable<oscPedestrianController *> oscPedestrianControllerMember;
typedef oscObjectVariableArray<oscPedestrianController *> oscPedestrianControllerArrayMember;


}

#endif //OSCPEDESTRIANCONTROLLER_H
