/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPARAMETERLIST_H
#define OSCPARAMETERLIST_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscParameter.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscParameterList : public oscObjectBase
{
public:
    oscParameterList()
    {
        OSC_OBJECT_ADD_MEMBER(Parameter, "oscParameter");
    };
    oscParameterMember Parameter;

};

typedef oscObjectVariable<oscParameterList *> oscParameterListMember;


}

#endif //OSCPARAMETERLIST_H
