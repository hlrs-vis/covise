/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PRIORITY_H
#define OSC_PRIORITY_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT priorityType: public oscEnumType
{
public:
    static priorityType *instance(); 
private:
    priorityType();
    static priorityType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPriority: public oscObjectBase
{
public:
    oscPriority()
    {
        OSC_ADD_MEMBER(priority);

        priority.enumType = priorityType::instance();
    };

    oscEnum priority;

    enum priority
    {
        overwrite,
        following,
        skip,
    };
};

typedef oscObjectVariable<oscPriority *> oscPriorityMember;

}

#endif //OSC_PRIORITY_H
