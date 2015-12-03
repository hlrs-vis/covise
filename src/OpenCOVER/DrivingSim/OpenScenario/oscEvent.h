/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_EVENT_H
#define OSC_EVENT_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscStartConditionGroup.h>
#include <oscAction.h>

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
class OPENSCENARIOEXPORT oscEvent: public oscObjectBase
{
public:
    oscEvent()
    {
        OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(priority);
		priority.enumType = priorityType::instance();
		OSC_OBJECT_ADD_MEMBER(startConditionGroup,"oscStartConditionGroup");
		OSC_OBJECT_ADD_MEMBER(action,"oscAction");
		
    };
	oscString name;
	enum priority
    {
        overwrite,
        following,
        skip,
    };
	oscEnum priority;
	oscStartConditionGroupMember startConditionGroup;
	oscActionMember action;
	
};

typedef oscObjectVariable<oscEvent *> oscEventMember;

}

#endif //OSC_EVENT_H
