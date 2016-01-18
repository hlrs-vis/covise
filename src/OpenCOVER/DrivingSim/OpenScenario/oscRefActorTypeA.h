/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFACTOR_TYPE_A_H
#define OSC_REFACTOR_TYPE_A_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscManeuverTypeCGroup.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorTypeA: public oscObjectBase
{
public:
    oscRefActorTypeA()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(maneuverGroup, "oscManeuverTypeCGroup");
    };

    oscString name;
    oscManeuverTypeCGroupArrayMember maneuverGroup;
};

typedef oscObjectVariable<oscRefActorTypeA *> oscRefActorTypeAMember;

}

#endif //OSC_REFACTOR_TYPE_A_H
