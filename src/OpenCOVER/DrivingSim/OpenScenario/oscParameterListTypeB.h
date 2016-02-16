/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PARAMETER_LIST_TYPE_B_H
#define OSC_PARAMETER_LIST_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscParameterTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscParameterListTypeB: public oscObjectBase
{
public:
    oscParameterListTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(parameter, "oscParameterTypeB");
    };

    oscParameterTypeBMember parameter;
};

typedef oscObjectVariableArray<oscParameterListTypeB *> oscParameterListTypeBMemberArray;

}

#endif /* OSC_PARAMETER_LIST_TYPE_B_H */
