/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PARAMETER_TYPE_A_LIST_H
#define OSC_PARAMETER_TYPE_A_LIST_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscParameterTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscParameterTypeAList: public oscObjectBase
{
public:
    oscParameterTypeAList()
    {
        OSC_OBJECT_ADD_MEMBER(parameter, "oscParameterTypeA");
    };

    oscParameterTypeAMember parameter;
};

typedef oscObjectArrayVariable<oscParameterTypeAList *> oscParameterTypeAListArrayMember;

}

#endif //OSC_PARAMETER_TYPE_A_LIST_H
