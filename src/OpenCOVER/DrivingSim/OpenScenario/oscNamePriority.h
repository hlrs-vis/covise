/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_NAME_PRIORITY_H
#define OSC_NAME_PRIORITY_H

#include "oscExport.h"
#include "oscNameUserData.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT priorityManeuverType: public oscEnumType
{
public:
    static priorityManeuverType *instance();
private:
    priorityManeuverType();
    static priorityManeuverType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNamePriority: public oscNameUserData
{
public:
    oscNamePriority()
    {
        OSC_ADD_MEMBER(priority);
        OSC_ADD_MEMBER(numberOfExecutions);

        priority.enumType = priorityManeuverType::instance();
    };

    oscEnum priority;
    oscInt numberOfExecutions;

    enum maneuverPriority
    {
        overwrite,
        following,
        cancel,
    };
};

typedef oscObjectVariable<oscNamePriority *> oscNamePriorityMember;

}

#endif //OSC_NAME_PRIORITY_H
