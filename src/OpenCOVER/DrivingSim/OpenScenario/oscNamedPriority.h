/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_NAMED_PRIORITY_H
#define OSC_NAMED_PRIORITY_H

#include "oscExport.h"
#include "oscNamedObject.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT maneuverPriorityType: public oscEnumType
{
public:
    static maneuverPriorityType *instance(); 
private:
    maneuverPriorityType();
    static maneuverPriorityType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNamedPriority: public oscNamedObject
{
public:
    oscNamedPriority()
    {   
        OSC_ADD_MEMBER(numberOfExecutions);
        OSC_ADD_MEMBER(priority);

        priority.enumType = maneuverPriorityType::instance();
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

typedef oscObjectVariable<oscNamedPriority *> osccNamedPriorityMember;

}

#endif //OSC_NAMED_PRIORITY_H
