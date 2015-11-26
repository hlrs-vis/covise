/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

// sort alphabetically
#include <oscAbsolute.h>
#include <oscAcceleration.h>
#include <oscAction.h>
#include <oscAutonomous.h>
#include <oscBehavior.h>
#include <oscBody.h>
#include <oscBoundingBox.h>
#include <oscCatalog.h>
#include <oscCatalogs.h>
#include <oscCenter.h>
#include <oscCollision.h>
#include <oscColor.h>
#include <oscCommand.h>
#include <oscCondition.h>
#include <oscCoord.h>
#include <oscControllerChoice.h>
#include <oscDate.h>
#include <oscDimension.h>
#include <oscDimensions.h>
#include <oscDirectory.h>
#include <oscDistance.h>
#include <oscDriver.h>
#include <oscDriverRef.h>
#include <oscDriverCatalog.h>
#include <oscEntity.h>
#include <oscEntityCatalog.h>
#include <oscEnvironment.h>
#include <oscFactories.h>
#include <oscFile.h>
#include <oscFilter.h>
#include <oscFog.h>
#include <oscFrustum.h>
#include <oscHeader.h>
#include <oscIntensity.h>
#include <oscLaneChange.h>
#include <oscLaneCoord.h>
#include <oscLaneDynamics.h>
#include <oscLight.h>
#include <oscManeuverCatalog.h>
#include <oscMiscObjectCatalog.h>
#include <oscMiscObjectRef.h>
#include <oscNamedObject.h>
#include <oscNameId.h>
#include <oscNumericCondition.h>
#include <oscObject.h>
#include <oscObjectBase.h>
#include <oscObjectChoice.h>
#include <oscObjectRef.h>
#include <oscObserver.h>
#include <oscObserverCatalog.h>
#include <oscObserverId.h>
#include <oscOffroad.h>
#include <oscOrientation.h>
#include <oscParameter.h>
#include <oscParameters.h>
#include <oscPartner.h>
#include <oscPedestrianCatalog.h>
#include <oscPedestrianController.h>
#include <oscPedestrianRef.h>
#include <oscPosition.h>
#include <oscPositionLane.h>
#include <oscPositionRoad.h>
#include <oscPositionRoute.h>
#include <oscPositionWorld.h>
#include <oscPositionXyz.h>
#include <oscPrecipitation.h>
#include <oscReachPosition.h>
#include <oscReferenceHanding.h>
#include <oscRelative.h>
#include <oscRelativeChoice.h>
#include <oscRelativePositionLane.h>
#include <oscRelativePositionRoad.h>
#include <oscRelativePositionWorld.h>
#include <oscRoadCondition.h>
#include <oscRoadConditions.h>
#include <oscRoadCoord.h>
#include <oscRoadNetwork.h>
#include <oscRoutingCatalog.h>
#include <oscSimulationTime.h>
#include <oscSpeed.h>
#include <oscSpeedDynamics.h>
#include <oscStandsStill.h>
#include <oscStartConditionGroup.h>
#include <oscStoppingDistance.h>
#include <oscTime.h>
#include <oscTest.h>
#include <oscTimeHeadway.h>
#include <oscTimeOfDay.h>
#include <oscTimeToCollision.h>
#include <oscUserData.h>
#include <oscUserDefined.h>
#include <oscVariables.h>
#include <oscVehicleCatalog.h>
#include <oscVehicleRef.h>
#include <oscVelocity.h>
#include <oscWeather.h>


using namespace OpenScenario;


oscFactory<oscObjectBase,std::string> staticObjectFactory;
oscFactory<oscMemberValue,oscMemberValue::MemberTypes> staticValueFactory;

oscFactories* oscFactories::inst = NULL;

oscFactories::oscFactories()
{
    // set default factories
    objectFactory = &staticObjectFactory;
    valueFactory = &staticValueFactory;

    // register all builtin types
    staticValueFactory.registerType<oscStringValue>(oscMemberValue::STRING);
    staticValueFactory.registerType<oscIntValue>(oscMemberValue::INT);
    staticValueFactory.registerType<oscUIntValue>(oscMemberValue::UINT);
    staticValueFactory.registerType<oscShortValue>(oscMemberValue::SHORT);
    staticValueFactory.registerType<oscUShortValue>(oscMemberValue::USHORT);
    staticValueFactory.registerType<oscDoubleValue>(oscMemberValue::DOUBLE);
    staticValueFactory.registerType<oscBoolValue>(oscMemberValue::BOOL);
    staticValueFactory.registerType<oscFloatValue>(oscMemberValue::FLOAT);
    staticValueFactory.registerType<oscEnumValue>(oscMemberValue::ENUM);
    
    // sort alphabetically
    staticObjectFactory.registerType<oscAbsolute>("oscAbsolute");
	staticObjectFactory.registerType<oscAcceleration>("oscAcceleration");
    staticObjectFactory.registerType<oscAction>("oscAction");
    staticObjectFactory.registerType<oscAutonomous>("oscAutonomous");
    staticObjectFactory.registerType<oscBehavior>("oscBehavior");
    staticObjectFactory.registerType<oscBody>("oscBody");
    staticObjectFactory.registerType<oscBoundingBox>("oscBoundingBox");
    staticObjectFactory.registerType<oscCatalog>("oscCatalog");
    staticObjectFactory.registerType<oscCatalogs>("oscCatalogs");
    staticObjectFactory.registerType<oscCenter>("oscCenter");
	staticObjectFactory.registerType<oscCollision>("oscCollision");
    staticObjectFactory.registerType<oscColor>("oscColor");
	staticObjectFactory.registerType<oscCommand>("oscCommand");
    staticObjectFactory.registerType<oscCondition>("oscCondition");
	staticObjectFactory.registerType<oscCoord>("oscCoord");
	staticObjectFactory.registerType<oscControllerChoice>("oscControllerChoice");
    staticObjectFactory.registerType<oscDate>("oscDate");
    staticObjectFactory.registerType<oscDimension>("oscDimension");
    staticObjectFactory.registerType<oscDimensions>("oscDimensions");
    staticObjectFactory.registerType<oscDirectory>("oscDirectory");
	staticObjectFactory.registerType<oscDistance>("oscDistance");
	staticObjectFactory.registerType<oscDriver>("oscDriver");
    staticObjectFactory.registerType<oscDriverCatalog>("oscDriverCatalog");
	staticObjectFactory.registerType<oscDriverRef>("oscDriverRef");
	staticObjectFactory.registerType<oscEntity>("oscEntity");
    staticObjectFactory.registerType<oscEntityCatalog>("oscEntityCatalog");
    staticObjectFactory.registerType<oscEnvironment>("oscEnvironment");
    staticObjectFactory.registerType<oscFile>("oscFile");
	staticObjectFactory.registerType<oscFilter>("oscFilter");
    staticObjectFactory.registerType<oscFog>("oscFog");
	staticObjectFactory.registerType<oscFrustum>("oscFrustum");
    staticObjectFactory.registerType<oscHeader>("oscHeader");
    staticObjectFactory.registerType<oscIntensity>("oscIntensity");
    staticObjectFactory.registerType<oscLaneChange>("oscLaneChange");
    staticObjectFactory.registerType<oscLaneCoord>("oscLaneCoord");
    staticObjectFactory.registerType<oscLaneDynamics>("oscLaneDynamics");
    staticObjectFactory.registerType<oscLight>("oscLight");
    staticObjectFactory.registerType<oscManeuverCatalog>("oscManeuverCatalog");
    staticObjectFactory.registerType<oscMiscObjectCatalog>("oscMiscObjectCatalog");
	staticObjectFactory.registerType<oscMiscObjectRef>("oscMiscObjectRef");
    staticObjectFactory.registerType<oscNamedObject>("oscNamedObject");
    staticObjectFactory.registerType<oscNameId>("oscNameId");
	staticObjectFactory.registerType<oscNumericCondition>("oscNumericCondition");
	staticObjectFactory.registerType<oscObject>("oscObject");
	staticObjectFactory.registerType<oscObjectChoice>("oscObjectChoice");
	staticObjectFactory.registerType<oscObjectRef>("oscObjectRef");
	staticObjectFactory.registerType<oscObserver>("oscObserver");
	staticObjectFactory.registerType<oscOffroad>("oscOffroad");
    staticObjectFactory.registerType<oscObserverCatalog>("oscObserverCatalog");
	staticObjectFactory.registerType<oscObserverId>("oscObserverId");
    staticObjectFactory.registerType<oscOrientation>("oscOrientation");
    staticObjectFactory.registerType<oscParameter>("oscParameter");
    staticObjectFactory.registerType<oscParameters>("oscParameters");
	staticObjectFactory.registerType<oscPartner>("oscPartner");
    staticObjectFactory.registerType<oscPedestrianCatalog>("oscPedestrianCatalog");
	staticObjectFactory.registerType<oscPedestrianController>("oscPedestrianController");
	staticObjectFactory.registerType<oscPedestrianRef>("oscPedestrianRef");
	staticObjectFactory.registerType<oscPosition>("oscPosition");
    staticObjectFactory.registerType<oscPositionLane>("oscPositionLane");
    staticObjectFactory.registerType<oscPositionRoad>("oscPositionRoad");
    staticObjectFactory.registerType<oscPositionRoute>("oscPositionRoute");
    staticObjectFactory.registerType<oscPositionWorld>("oscPositionWorld");
    staticObjectFactory.registerType<oscPositionXyz>("oscPositionXyz");
    staticObjectFactory.registerType<oscPrecipitation>("oscPrecipitation");
	staticObjectFactory.registerType<oscReachPosition>("oscReachPosition");
	staticObjectFactory.registerType<oscReferenceHanding>("oscReferenceHanding");
    staticObjectFactory.registerType<oscRelative>("oscRelative");
    staticObjectFactory.registerType<oscRelativeChoice>("oscRelativeChoice");
    staticObjectFactory.registerType<oscRelativePositionLane>("oscRelativePositionLane");
    staticObjectFactory.registerType<oscRelativePositionRoad>("oscRelativePositionRoad");
    staticObjectFactory.registerType<oscRelativePositionWorld>("oscRelativePositionWorld");
    staticObjectFactory.registerType<oscRoadCondition>("oscRoadCondition");
    staticObjectFactory.registerType<oscRoadConditions>("oscRoadConditions");
    staticObjectFactory.registerType<oscRoadCoord>("oscRoadCoord");
    staticObjectFactory.registerType<oscRoadNetwork>("oscRoadNetwork");
    staticObjectFactory.registerType<oscRoutingCatalog>("oscRoutingCatalog");
	staticObjectFactory.registerType<oscSimulationTime>("oscSimulationTime");
    staticObjectFactory.registerType<oscSpeed>("oscSpeed");
    staticObjectFactory.registerType<oscSpeedDynamics>("oscSpeedDynamics");
	staticObjectFactory.registerType<oscStandsStill>("oscStandsStill");
    staticObjectFactory.registerType<oscStartConditionGroup>("oscStartConditionGroup");
	staticObjectFactory.registerType<oscStoppingDistance>("oscStoppingDistance");
    staticObjectFactory.registerType<oscTime>("oscTime");
    staticObjectFactory.registerType<oscTest>("oscTest");
	staticObjectFactory.registerType<oscObject>("oscObject");
    staticObjectFactory.registerType<oscTimeHeadway>("oscTimeHeadway");
	staticObjectFactory.registerType<oscTimeOfDay>("oscTimeOfDay");
	staticObjectFactory.registerType<oscTimeToCollision>("oscTimeToCollision");
    staticObjectFactory.registerType<oscUserData>("oscUserData");
	staticObjectFactory.registerType<oscUserDefined>("oscUserDefined");
    staticObjectFactory.registerType<oscVehicleCatalog>("oscVehicleCatalog");
	staticObjectFactory.registerType<oscVehicleRef>("oscVehicleRef");
	staticObjectFactory.registerType<oscVelocity>("oscVelocity");
    staticObjectFactory.registerType<oscWeather>("oscWeather");
//    staticObjectFactory.registerType<>("");
}


void oscFactories::setObjectFactory(oscFactory<oscObjectBase,std::string> *f)
{
    objectFactory = f;
    if(objectFactory == NULL)
        objectFactory = &staticObjectFactory;
}

void oscFactories::setValueFactory(oscFactory<oscMemberValue,oscMemberValue::MemberTypes> *f)
{
    valueFactory = f;
    if(valueFactory == NULL)
        valueFactory = &staticValueFactory;
}
