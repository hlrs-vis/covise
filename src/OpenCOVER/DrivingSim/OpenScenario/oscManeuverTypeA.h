/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_A_H
#define OSC_MANEUVER_TYPE_A_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscHeader.h>
#include <oscEvents.h>
#include <oscParameterTypeAList.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeA: public oscObjectBase
{
public:
    oscManeuverTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(header, "oscHeader");
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(parameterList, "oscParameterTypeAList");
        OSC_OBJECT_ADD_MEMBER(events, "oscEvents");
    };

    oscHeaderMember header;
    oscString name;
    oscParameterTypeAListArrayMember parameterList;
    oscEventsArrayMember events;
};

typedef oscObjectVariable<oscManeuverTypeA *>oscManeuverTypeAMember;

}

#endif //OSC_MANEUVER_TYPE_A_H
