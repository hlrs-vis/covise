/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ACTION_H
#define OSC_ACTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscAutonomous.h"
#include "oscSpeed.h"
#include "oscLaneChange.h"
#include "oscFollowRoute.h"
#include "oscLaneOffset.h"
#include "oscPosition.h"
#include "oscDistanceLateral.h"
#include "oscDistanceLongitudinal.h"
#include "oscVisibility.h"
#include "oscCharacterAppearance.h"
#include "oscCharacterMotion.h"
#include "oscCharacterGesture.h"
#include "oscTrafficLight.h"
#include "oscEntityAdd.h"
#include "oscEntityDelete.h"
#include "oscTrafficJam.h"
#include "oscTrafficSource.h"
#include "oscTrafficSink.h"
#include "oscUserDefinedCommand.h"
#include "oscUserDefinedAction.h"
#include "oscUserScript.h"
#include "oscNotify.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAction: public oscObjectBase
{
public:
    oscAction()
    {
        OSC_OBJECT_ADD_MEMBER_CHOICE(autonomous, "oscAutonomous");
        OSC_OBJECT_ADD_MEMBER_CHOICE(speed, "oscSpeed");
        OSC_OBJECT_ADD_MEMBER_CHOICE(laneChange, "oscLaneChange");
        OSC_OBJECT_ADD_MEMBER_CHOICE(followRoute, "oscFollowRoute");
        OSC_OBJECT_ADD_MEMBER_CHOICE(laneOffset, "oscLaneOffset");
        OSC_OBJECT_ADD_MEMBER_CHOICE(position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER_CHOICE(distanceLateral, "oscDistanceLateral");
        OSC_OBJECT_ADD_MEMBER_CHOICE(distanceLongitudinal, "oscDistanceLongitudinal");
        OSC_OBJECT_ADD_MEMBER_CHOICE(visibility, "oscVisibility");
        OSC_OBJECT_ADD_MEMBER_CHOICE(characterAppearance, "oscCharacterAppearance");
        OSC_OBJECT_ADD_MEMBER_CHOICE(characterMotion, "oscCharacterMotion");
        OSC_OBJECT_ADD_MEMBER_CHOICE(characterGesture, "oscCharacterGesture");
        OSC_OBJECT_ADD_MEMBER_CHOICE(trafficLight, "oscTrafficLight");
        OSC_OBJECT_ADD_MEMBER_CHOICE(entityAdd, "oscEntityAdd");
        OSC_OBJECT_ADD_MEMBER_CHOICE(entityDelete, "oscEntityDelete");
        OSC_OBJECT_ADD_MEMBER_CHOICE(trafficJam, "oscTrafficJam");
        OSC_OBJECT_ADD_MEMBER_CHOICE(trafficSource, "oscTrafficSource");
        OSC_OBJECT_ADD_MEMBER_CHOICE(trafficSink, "oscTrafficSink");
        OSC_OBJECT_ADD_MEMBER_CHOICE(userDefinedCommand, "oscUserDefinedCommand");
        OSC_OBJECT_ADD_MEMBER_CHOICE(userDefinedAction, "oscUserDefinedAction");
        OSC_OBJECT_ADD_MEMBER_CHOICE(userScript, "oscUserScript");
        OSC_OBJECT_ADD_MEMBER_CHOICE(notify, "oscNotify");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    };

    oscAutonomousMember autonomous;
    oscSpeedMember speed;
    oscLaneChangeMember laneChange;
    oscFollowRouteMember followRoute;
    oscLaneOffsetMember laneOffset;
    oscPositionMember position;
    oscDistanceLateralMember distanceLateral;
    oscDistanceLongitudinalMember distanceLongitudinal;
    oscVisibilityMember visibility;
    oscCharacterAppearanceMember characterAppearance;
    oscCharacterMotionMember characterMotion;
    oscCharacterGestureMember characterGesture;
    oscTrafficLightMember trafficLight;
    oscEntityAddMember entityAdd;
    oscEntityDeleteMember entityDelete;
    oscTrafficJamMember trafficJam;
    oscTrafficSourceMember trafficSource;
    oscTrafficSinkMember trafficSink;
    oscUserDefinedCommandMember userDefinedCommand;
    oscUserDefinedActionMember userDefinedAction;
    oscUserScriptMember userScript;
    oscNotifyMember notify;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscAction *> oscActionMember;

}

#endif //OSC_ACTION_H
