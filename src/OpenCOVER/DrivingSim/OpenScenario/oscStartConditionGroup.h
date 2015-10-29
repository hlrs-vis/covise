/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_START_CONGDITION_GROUP_H
#define OSC_START_CONGDITION_GROUP_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionGroup: public oscObjectBase
{
public:
    oscStartConditionGroup()
    {
        OSC_ADD_MEMBER(URL);
    };
    oscString URL;
};

typedef oscObjectVariable<oscStartConditionGroup *> oscStartConditionGroupMember;

}

#endif //OSC_START_CONGDITION_GROUP_H