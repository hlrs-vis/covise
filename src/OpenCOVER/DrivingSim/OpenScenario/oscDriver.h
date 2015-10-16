/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_DRIVER_H
#define OSC_DRIVER_H
#include <oscExport.h>
#include <oscFile.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscHeader.h>
#include <oscFile.h>
#include <oscBody.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscDriver: public oscObjectBase
{
public:
    oscDriver()
    {
        OSC_ADD_MEMBER(header);
        OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(obeyTrafficLights);
        OSC_ADD_MEMBER(obeyTrafficSigns);
		OSC_ADD_MEMBER(steeringDistance);
        OSC_ADD_MEMBER(foresightDistance);
		OSC_ADD_MEMBER(respondToTailgating);
        OSC_ADD_MEMBER(urgeToOvertake);
		OSC_ADD_MEMBER(useOfIndicator);
        OSC_ADD_MEMBER(keepRightRule);
		OSC_ADD_MEMBER(laneChangeDynamic);
        OSC_ADD_MEMBER(speedKeeping);
		OSC_ADD_MEMBER(laneKeeping);
		OSC_ADD_MEMBER(distanceKeeping);
        OSC_ADD_MEMBER(observeSpeedLimits);
		OSC_ADD_MEMBER(curveSpeed);
        OSC_ADD_MEMBER(desiredDeceleration);
		OSC_ADD_MEMBER(desiredAcceleration);
        OSC_ADD_MEMBER(desiredVelocity);
		OSC_ADD_MEMBER(politeness);
		OSC_ADD_MEMBER(alertness);
        OSC_ADD_MEMBER(adaptToVehicleType);
		OSC_ADD_MEMBER(adaptToTimeOfDay);
        OSC_ADD_MEMBER(adaptToRoadConditions);
		OSC_ADD_MEMBER(adaptToWeatherConditions);
        OSC_ADD_MEMBER(body);
		OSC_ADD_MEMBER(include);
        OSC_ADD_MEMBER(geometry);
    };
    oscHeaderMember header;
    oscNameMember name;
	oscBool obeyTrafficLights;
	oscBool obeyTrafficSigns;
	oscDouble steeringDistance;
	oscDouble foresightDistance;
	oscDouble respondToTailgating;
	oscDouble urgeToOvertake;
	oscDouble useOfIndicator;
	oscDouble keepRightRule;
	oscDouble laneChangeDynamic;
	oscDouble speedKeeping;
	oscDouble laneKeeping;
	oscDouble distanceKeeping;
	oscDouble observeSpeedLimits;
	oscDouble curveSpeed;
	oscDouble desiredDeceleration;
	oscDouble desiredAcceleration;
	oscDouble desiredVelocity;
	oscDouble politeness;
	oscDouble alertness;
	oscDouble adaptToVehicleType;
	oscDouble adaptToTimeOfDay;
	oscDouble adaptToRoadConditions;
	oscDouble adaptToWeatherConditions;
	oscBodyMember body;
	oscFileMember include;
	oscFileMember geometry;
};

typedef oscObjectVariable<oscDriver *> oscDriverMember;

}

#endif //OSC_DRIVER_H