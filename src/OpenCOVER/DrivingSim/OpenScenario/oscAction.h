/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ACTION_H
#define OSC_ACTION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscNamedObject.h>
#include <oscAutonomous.h>
#include <oscSpeed.h>
#include <oscLaneChange.h>
#include <oscLaneOffset.h>
#include <oscPosition.h>
#include <oscDistanceLateral.h>
#include <oscDistanceLongitudinal.h>
#include <oscVisibility.h>
#include <oscCharacterAppearance.h>
#include <oscCharacterGesture.h>
#include <oscCharacterMotion.h>
#include <oscTrafficLight.h>
#include <oscUserData.h>
#include <oscFile.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAction: public oscNamedObject
{
public:
    oscAction()
    {
		OSC_OBJECT_ADD_MEMBER(autonomous,"oscAutonomous");
		OSC_OBJECT_ADD_MEMBER(speed,"oscSpeed");
		OSC_OBJECT_ADD_MEMBER(laneChange,"oscLaneChange");
		OSC_OBJECT_ADD_MEMBER(laneOffset,"oscLaneOffset");
		OSC_OBJECT_ADD_MEMBER(position,"oscPosition");
		OSC_OBJECT_ADD_MEMBER(distanceLateral,"oscDistanceLateral");
		OSC_OBJECT_ADD_MEMBER(distanceLongitudinal,"oscDistanceLongitudinal");
		OSC_OBJECT_ADD_MEMBER(visibility,"oscVisibility");
	    OSC_OBJECT_ADD_MEMBER(characterAppearance,"oscCharacterAppearance");
		OSC_OBJECT_ADD_MEMBER(characterGesture,"oscCharacterGesture");
		OSC_OBJECT_ADD_MEMBER(characterMotion,"oscCharacterMotion");
		OSC_OBJECT_ADD_MEMBER(trafficLight,"oscTrafficLight");
		OSC_OBJECT_ADD_MEMBER(userData,"oscUserData");
		OSC_OBJECT_ADD_MEMBER(include,"oscFile");
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
	oscUserDataMember userData;
	oscFileMember iclude;
};

typedef oscObjectVariable<oscAction *> oscActionMember;

}

#endif //OSC_ACTION_H
