/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ACTION_H
#define OSC_ACTION_H

#include "oscExport.h"
#include "oscNameUserData.h"
#include "oscObjectVariable.h"

#include "oscAutonomous.h"
#include "oscSpeed.h"
#include "oscLaneChange.h"
#include "oscLaneOffset.h"
#include "oscPosition.h"
#include "oscDistanceLateral.h"
#include "oscDistanceLongitudinal.h"
#include "oscVisibility.h"
#include "oscCharacterAppearance.h"
#include "oscCharacterGesture.h"
#include "oscCharacterMotion.h"
#include "oscTrafficLight.h"
#include "oscEntityAdd.h"
#include "oscEntityDelete.h"
#include "oscTrafficJam.h"
#include "oscTrafficSource.h"
#include "oscTrafficSink.h"
#include "oscUserDefinedAction.h"
#include "oscUserDefinedCommand.h"
#include "oscNotify.h"
#include "oscUserScript.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAction: public oscNameUserData
{
public:
    oscAction()
    {
        OSC_OBJECT_ADD_MEMBER(autonomous, "oscAutonomous");
        OSC_OBJECT_ADD_MEMBER(speed, "oscSpeed");
        OSC_OBJECT_ADD_MEMBER(laneChange, "oscLaneChange");
        OSC_OBJECT_ADD_MEMBER(laneOffset, "oscLaneOffset");
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(distanceLateral, "oscDistanceLateral");
        OSC_OBJECT_ADD_MEMBER(distanceLongitudinal, "oscDistanceLongitudinal");
        OSC_OBJECT_ADD_MEMBER(visibility, "oscVisibility");
        OSC_OBJECT_ADD_MEMBER(characterAppearance, "oscCharacterAppearance");
        OSC_OBJECT_ADD_MEMBER(characterGesture, "oscCharacterGesture");
        OSC_OBJECT_ADD_MEMBER(characterMotion, "oscCharacterMotion");
        OSC_OBJECT_ADD_MEMBER(trafficLight, "oscTrafficLight");
        OSC_OBJECT_ADD_MEMBER(entityAdd, "oscEntityAdd");
        OSC_OBJECT_ADD_MEMBER(entityDelete, "oscEntityDelete");
        OSC_OBJECT_ADD_MEMBER(trafficJam, "oscTrafficJam");
        OSC_OBJECT_ADD_MEMBER(trafficSource, "oscTrafficSource");
        OSC_OBJECT_ADD_MEMBER(trafficSink, "oscTrafficSink");
        OSC_OBJECT_ADD_MEMBER(userDefinedAction, "oscUserDefinedAction");
        OSC_OBJECT_ADD_MEMBER(userDefinedCommand, "oscUserDefinedCommand");
        OSC_OBJECT_ADD_MEMBER(userScript, "oscUserScript");
        OSC_OBJECT_ADD_MEMBER(notify, "oscNotify");
        OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

    oscAutonomousMember autonomous;
    oscSpeedMember speed;
    oscLaneChangeMember laneChange;
    oscLaneOffsetMember laneOffset;
    oscPositionMember position;
    oscDistanceLateralMember distanceLateral;
    oscDistanceLongitudinalMember distanceLongitudinal;
    oscVisibilityMember visibility;
    oscCharacterAppearanceMember characterAppearance;
    oscCharacterGestureMember characterGesture;
    oscCharacterMotionMember characterMotion;
    oscTrafficLightMember trafficLight;
    oscEntityAddMember entityAdd;
    oscEntityDeleteMember entityDelete;
    oscTrafficJamMember trafficJam;
    oscTrafficSourceMember trafficSource;
    oscTrafficSinkMember trafficSink;
    oscUserDefinedActionMember userDefinedAction;
    oscUserDefinedCommandMember userDefinedCommand;
    oscUserScriptMember userScript;
    oscNotifyMember notify;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscAction *> oscActionMember;

}

#endif //OSC_ACTION_H
