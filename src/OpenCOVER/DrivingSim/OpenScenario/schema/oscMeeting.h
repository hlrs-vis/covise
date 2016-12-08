/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMEETING_H
#define OSCMEETING_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Meeting_Position_modeType : public oscEnumType
{
public:
static Enum_Meeting_Position_modeType *instance();
    private:
		Enum_Meeting_Position_modeType();
	    static Enum_Meeting_Position_modeType *inst; 
};
class OPENSCENARIOEXPORT oscMeeting : public oscObjectBase
{
public:
oscMeeting()
{
        OSC_ADD_MEMBER(mode);
        OSC_ADD_MEMBER(timingOffset);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition");
        mode.enumType = Enum_Meeting_Position_modeType::instance();
    };
    oscEnum mode;
    oscDouble timingOffset;
    oscPositionMember Position;

    enum Enum_Meeting_Position_mode
    {
straight,
route,

    };

};

typedef oscObjectVariable<oscMeeting *> oscMeetingMember;
typedef oscObjectVariableArray<oscMeeting *> oscMeetingArrayMember;


}

#endif //OSCMEETING_H
