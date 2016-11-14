/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PRIVATE_H
#define OSC_PRIVATE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscParameterListTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPrivate: public oscObjectBase
{
public:
    oscPrivate()
    {	
        OSC_ADD_MEMBER(uid);
        OSC_OBJECT_ADD_MEMBER(parameterList, "oscParameterListTypeB");
    };

    oscString uid;
    oscParameterListTypeBArrayMember parameterList;
};

typedef oscObjectVariable<oscPrivate *> oscPrivateMember;

}

#endif //OSC_PRIVATE_H
