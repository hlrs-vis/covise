/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PARAMETER_LIST_H
#define OSC_PARAMETER_LIST_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscVariables.h>
#include <oscParameters.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscParameterList: public oscObjectBase
{
public:
    oscParameterList()
    {
        OSC_OBJECT_ADD_MEMBER(parameter,"oscParameters");
    };

    oscParametersMember parameter;
};

typedef oscObjectArrayVariable<oscParameterList *> oscParameterListArrayMember;

}

#endif //OSC_PARAMETER_LIST_H
