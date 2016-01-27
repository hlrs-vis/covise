/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFERENCE_HANDLING_H
#define OSC_REFERENCE_HANDLING_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT refHandlConditionType: public oscEnumType
{
public:
    static refHandlConditionType *instance();
private:
    refHandlConditionType();
    static refHandlConditionType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscReferenceHandling: public oscObjectBase
{
public:
    oscReferenceHandling()
    {
        OSC_ADD_MEMBER(maneuverId);
        OSC_ADD_MEMBER(eventId);
        OSC_ADD_MEMBER(condition);

        condition.enumType = refHandlConditionType::instance();
    };

    oscInt maneuverId;
    oscInt eventId;
    oscEnum condition;

    enum refHandlCondition
    {
        starts,
        ends,
        cancels,
    };
};

typedef oscObjectVariable<oscReferenceHandling *> oscReferenceHandlingMember;

}

#endif //OSC_REFERENCE_HANDLING_H
