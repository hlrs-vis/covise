/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_USER_DEFINED_H
#define OSC_USER_DEFINED_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscParameter.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscUserDefined: public oscObjectBase
{
public:
    oscUserDefined()
    {	
		OSC_ADD_MEMBER(uid);
		OSC_OBJECT_ADD_MEMBER(parameter, "oscParameter");
    };
	oscString uid;
	oscParameterMember parameter;
};

typedef oscObjectVariable<oscUserDefined *> oscUserDefinedMember;

}

#endif //OSC_USER_DEFINED_H
