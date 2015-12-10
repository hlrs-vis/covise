/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

// sort alphabetically
#include <oscAbsolute.h>
#include <oscAcceleration.h>
#include <oscAction.h>
#include <oscAerodynamics.h>
#include <oscAutonomous.h>
#include <oscBehavior.h>
#include <oscBody.h>
#include <oscBoundingBox.h>
#include <oscCancelCondition.h>
#include <oscCancelConditionGroup.h>
#include <oscCancelConditionRef.h>
#include <oscCatalogRef.h>
#include <oscCatalogs.h>
#include <oscCenter.h>
#include <oscCharacterAppearance.h>
#include <oscCharacterGesture.h>
#include <oscCharacterMotion.h>
#include <oscClothoid.h>
#include <oscCog.h>
#include <oscCollision.h>
#include <oscColor.h>
#include <oscCommand.h>
#include <oscCondition.h>
#include <oscConditionChoiceObject.h>
#include <oscConditionObject.h>
#include <oscContinuation.h>
#include <oscCoord.h>
#include <oscControllerChoice.h>
#include <oscDate.h>
#include <oscDimension.h>
#include <oscDimensions.h>
#include <oscDirectory.h>
#include <oscDistance.h>
#include <oscDistanceLateral.h>
#include <oscDistanceLongitudinal.h>
#include <oscDriver.h>
#include <oscDriverRef.h>
#include <oscDriverCatalog.h>
#include <oscEndCondition.h>
#include <oscEndConditionGroup.h>
#include <oscEndConditionRef.h>
#include <oscEngine.h>
#include <oscEntities.h>
#include <oscEntity.h>
#include <oscEntityAdd.h>
#include <oscEntityCatalog.h>
#include <oscEntityDelete.h>
#include <oscEnvironment.h>
#include <oscEnvironmentCatalog.h>
#include <oscEyepoint.h>
#include <oscEyepoints.h>
#include <oscEvent.h>
#include <oscEventStartCondition.h>
#include <oscEventStartConditionGroup.h>
#include <oscFactories.h>
#include <oscFeature.h>
#include <oscFeatures.h>
#include <oscFile.h>
#include <oscFilter.h>
#include <oscFog.h>
#include <oscFrustum.h>
#include <oscGearbox.h>
#include <oscGeneral.h>
#include <oscHeader.h>
#include <oscInitDynamics.h>
#include <oscInitPosition.h>
#include <oscInitState.h>
#include <oscIntensity.h>
#include <oscLaneChange.h>
#include <oscLaneCoord.h>
#include <oscLaneDynamics.h>
#include <oscLaneOffset.h>
#include <oscLaneOffsetDynamics.h>
#include <oscLight.h>
#include <oscLighting.h>
#include <oscLights.h>
#include <oscManeuver.h>
#include <oscManeuverCatalog.h>
#include <oscManeuverList.h>
#include <oscManeuverRef.h>
#include <oscManeuverRefActor.h>
#include <oscMirror.h>
#include <oscMirrors.h>
#include <oscMiscObjectCatalog.h>
#include <oscMiscObjectRef.h>
#include <oscNamedObject.h>
#include <oscNamedPriority.h>
#include <oscNameRefId.h>
#include <oscNotify.h>
#include <oscNumericCondition.h>
#include <oscObject.h>
#include <oscObjectBase.h>
#include <oscObjectChoice.h>
#include <oscConditionChoiceRefObject.h>
#include <oscObserver.h>
#include <oscObserverCatalog.h>
#include <oscObserverId.h>
#include <oscOffroad.h>
#include <oscOrientation.h>
#include <oscParameter.h>
#include <oscParameterList.h>
#include <oscParameters.h>
#include <oscPartner.h>
#include <oscPedestrian.h>
#include <oscPedestrianCatalog.h>
#include <oscPedestrianController.h>
#include <oscPedestrianRef.h>
#include <oscPerformance.h>
#include <oscPosition.h>
#include <oscPositionLane.h>
#include <oscPositionRoad.h>
#include <oscPositionRoute.h>
#include <oscPositionWorld.h>
#include <oscPositionXyz.h>
#include <oscPrecipitation.h>
#include <oscPriority.h>
#include <oscReachPosition.h>
#include <oscRefActor.h>
#include <oscRefActorList.h>
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
#include <oscRoute.h>
#include <oscRouting.h>
#include <oscRoutingCatalog.h>
#include <oscScenarioEnd.h>
#include <oscSimulationTime.h>
#include <oscShape.h>
#include <oscSpeed.h>
#include <oscSpeedDynamics.h>
#include <oscSetState.h>
#include <oscSetController.h>
#include <oscSpline.h>
#include <oscStandsStill.h>
#include <oscStartCondition.h>
#include <oscStartConditionGroup.h>
#include <oscStartConditionRef.h>
#include <oscStoppingDistance.h>
#include <oscStoryboard.h>
#include <oscTime.h>
#include <oscTest.h>
#include <oscTimeHeadway.h>
#include <oscTimeOfDay.h>
#include <oscTimeToCollision.h>
#include <oscTrafficJam.h>
#include <oscTrafficLight.h>
#include <oscTrafficSink.h>
#include <oscTrafficSource.h>
#include <oscUserData.h>
#include <oscUserDefined.h>
#include <oscUserDefinedAction.h>
#include <oscUserDefinedCommand.h>
#include <oscUserScript.h>
#include <oscVariables.h>
#include <oscVehicle.h>
#include <oscVehicleAxle.h>
#include <oscVehicleCatalog.h>
#include <oscVehicleRef.h>
#include <oscVelocity.h>
#include <oscVisibility.h>
#include <oscWaypoint.h>
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
	staticObjectFactory.registerType<oscAerodynamics>("oscAerodynamics");
    staticObjectFactory.registerType<oscAutonomous>("oscAutonomous");
	staticObjectFactory.registerType<oscAxles>("oscAxles");
    staticObjectFactory.registerType<oscBehavior>("oscBehavior");
    staticObjectFactory.registerType<oscBody>("oscBody");
    staticObjectFactory.registerType<oscBoundingBox>("oscBoundingBox");
	staticObjectFactory.registerType<oscCancelCondition>("oscCancelCondition");
	staticObjectFactory.registerType<oscCancelConditionGroup>("oscCancelConditionGroup");
	staticObjectFactory.registerType<oscCancelConditionRef>("oscCancelConditionRef");
	staticObjectFactory.registerType<oscCatalogRef>("oscCatalogRef");
    staticObjectFactory.registerType<oscCatalogs>("oscCatalogs");
    staticObjectFactory.registerType<oscCenter>("oscCenter");
	staticObjectFactory.registerType<oscCharacterAppearance>("oscCharacterAppearance");
	staticObjectFactory.registerType<oscCharacterGesture>("oscCharacterGesture");
	staticObjectFactory.registerType<oscCharacterMotion>("oscCharacterMotion");
	staticObjectFactory.registerType<oscClothoid>("oscClothoid");
	staticObjectFactory.registerType<oscCog>("oscCog");
	staticObjectFactory.registerType<oscCollision>("oscCollision");
    staticObjectFactory.registerType<oscColor>("oscColor");
	staticObjectFactory.registerType<oscCommand>("oscCommand");
    staticObjectFactory.registerType<oscCondition>("oscCondition");
	staticObjectFactory.registerType<oscConditionChoiceRefObject>("oscConditionChoiceRefObject");
	staticObjectFactory.registerType<oscConditionObject>("oscConditionObject");
	staticObjectFactory.registerType<oscContinuation>("oscContinuation");
	staticObjectFactory.registerType<oscCoord>("oscCoord");
	staticObjectFactory.registerType<oscControllerChoice>("oscControllerChoice");
    staticObjectFactory.registerType<oscDate>("oscDate");
    staticObjectFactory.registerType<oscDimension>("oscDimension");
    staticObjectFactory.registerType<oscDimensions>("oscDimensions");
    staticObjectFactory.registerType<oscDirectory>("oscDirectory");
	staticObjectFactory.registerType<oscDistance>("oscDistance"); 
	staticObjectFactory.registerType<oscDistanceLateral>("oscDistanceLateral");
	staticObjectFactory.registerType<oscDistanceLongitudinal>("oscDistanceLongitudinal");
	staticObjectFactory.registerType<oscDriver>("oscDriver");
    staticObjectFactory.registerType<oscDriverCatalog>("oscDriverCatalog");
	staticObjectFactory.registerType<oscDriverRef>("oscDriverRef");
	staticObjectFactory.registerType<oscEndCondition>("oscEndCondition");
	staticObjectFactory.registerType<oscEndConditionGroup>("oscEndConditionGroup");
	staticObjectFactory.registerType<oscEndConditionRef>("oscEndConditionRef");
	staticObjectFactory.registerType<oscEngine>("oscEngine");
	staticObjectFactory.registerType<oscEntities>("oscEntities");
	staticObjectFactory.registerType<oscEntity>("oscEntity");
	staticObjectFactory.registerType<oscEntityAdd>("oscEntityAdd");
    staticObjectFactory.registerType<oscEntityCatalog>("oscEntityCatalog");
	staticObjectFactory.registerType<oscEntityDelete>("oscEntityDelete");
    staticObjectFactory.registerType<oscEnvironment>("oscEnvironment");
	staticObjectFactory.registerType<oscEnvironmentCatalog>("oscEnvironmentCatalog");
	staticObjectFactory.registerType<oscEyepoint>("oscEyepoint");
	staticObjectFactory.registerType<oscEyepoints>("oscEyepoints");
	staticObjectFactory.registerType<oscEvent>("oscEvent");
	staticObjectFactory.registerType<oscEventStartCondition>("oscEventStartCondition");
	staticObjectFactory.registerType<oscEventStartConditionGroup>("oscEventStartConditionGroup");
	staticObjectFactory.registerType<oscFeature>("oscFeature");
	staticObjectFactory.registerType<oscFeatures>("oscFeatures");
    staticObjectFactory.registerType<oscFile>("oscFile");
	staticObjectFactory.registerType<oscFilter>("oscFilter");
    staticObjectFactory.registerType<oscFog>("oscFog");
	staticObjectFactory.registerType<oscFrustum>("oscFrustum");
	staticObjectFactory.registerType<oscGearbox>("oscGearbox");
	staticObjectFactory.registerType<oscGeneral>("oscGeneral");
    staticObjectFactory.registerType<oscHeader>("oscHeader");
	staticObjectFactory.registerType<oscInitDynamics>("oscInitDynamics");
	staticObjectFactory.registerType<oscInitPosition>("oscInitPosition");
	staticObjectFactory.registerType<oscInitState>("oscInitState");
    staticObjectFactory.registerType<oscIntensity>("oscIntensity");
    staticObjectFactory.registerType<oscLaneChange>("oscLaneChange");
    staticObjectFactory.registerType<oscLaneCoord>("oscLaneCoord");
    staticObjectFactory.registerType<oscLaneDynamics>("oscLaneDynamics");
	staticObjectFactory.registerType<oscLaneOffset>("oscLaneOffset");
	staticObjectFactory.registerType<oscLaneOffsetDynamics>("oscLaneOffsetDynamics");
    staticObjectFactory.registerType<oscLight>("oscLight");
	staticObjectFactory.registerType<oscLighting>("oscLighting");
	staticObjectFactory.registerType<oscLights>("oscLights");
	staticObjectFactory.registerType<oscManeuver>("oscManeuver");
	staticObjectFactory.registerType<oscManeuverList>("oscManeuverList");
    staticObjectFactory.registerType<oscManeuverCatalog>("oscManeuverCatalog");
	staticObjectFactory.registerType<oscManeuverRef>("oscManeuverRef");
	staticObjectFactory.registerType<oscManeuverRefActor>("oscManeuverRefActor");
	staticObjectFactory.registerType<oscMirror>("oscMirror");
	staticObjectFactory.registerType<oscMirrors>("oscMirrors");
    staticObjectFactory.registerType<oscMiscObjectCatalog>("oscMiscObjectCatalog");
	staticObjectFactory.registerType<oscMiscObjectRef>("oscMiscObjectRef");
    staticObjectFactory.registerType<oscNamedObject>("oscNamedObject");
    staticObjectFactory.registerType<oscNameRefId>("oscNameRefId");
	staticObjectFactory.registerType<oscNamedPriority>("oscNamedPriority");
	staticObjectFactory.registerType<oscNotify>("oscNotify");
	staticObjectFactory.registerType<oscNumericCondition>("oscNumericCondition");
	staticObjectFactory.registerType<oscConditionChoiceObject>("oscConditionChoiceObject");
	staticObjectFactory.registerType<oscObjectChoice>("oscObjectChoice");
	staticObjectFactory.registerType<oscObject>("oscObject");
	staticObjectFactory.registerType<oscObserver>("oscObserver");
    staticObjectFactory.registerType<oscObserverCatalog>("oscObserverCatalog");
	staticObjectFactory.registerType<oscObserverId>("oscObserverId");
	staticObjectFactory.registerType<oscOffroad>("oscOffroad");
    staticObjectFactory.registerType<oscOrientation>("oscOrientation");
    staticObjectFactory.registerType<oscParameter>("oscParameter");
	staticObjectFactory.registerType<oscParameterList>("oscParameterList");
    staticObjectFactory.registerType<oscParameters>("oscParameters");
	staticObjectFactory.registerType<oscPartner>("oscPartner");
    staticObjectFactory.registerType<oscPedestrianCatalog>("oscPedestrianCatalog");
	staticObjectFactory.registerType<oscPedestrian>("oscPedestrian");
	staticObjectFactory.registerType<oscPedestrianRef>("oscPedestrianRef");
	staticObjectFactory.registerType<oscPedestrianController>("oscPedestrianController");
	staticObjectFactory.registerType<oscPerformance>("oscPerformance"); 
	staticObjectFactory.registerType<oscPosition>("oscPosition");
    staticObjectFactory.registerType<oscPositionLane>("oscPositionLane");
    staticObjectFactory.registerType<oscPositionRoad>("oscPositionRoad");
    staticObjectFactory.registerType<oscPositionRoute>("oscPositionRoute");
    staticObjectFactory.registerType<oscPositionWorld>("oscPositionWorld");
    staticObjectFactory.registerType<oscPositionXyz>("oscPositionXyz");
    staticObjectFactory.registerType<oscPrecipitation>("oscPrecipitation");
	staticObjectFactory.registerType<oscPriority>("oscPriority");
	staticObjectFactory.registerType<oscReachPosition>("oscReachPosition");
	staticObjectFactory.registerType<oscRefActor>("oscRefActor");
	staticObjectFactory.registerType<oscRefActorList>("oscRefActorList");
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
	staticObjectFactory.registerType<oscRoute>("oscRoute");
	staticObjectFactory.registerType<oscRouting>("oscRouting");
    staticObjectFactory.registerType<oscRoutingCatalog>("oscRoutingCatalog");
	staticObjectFactory.registerType<oscScenarioEnd>("oscScenarioEnd");
	staticObjectFactory.registerType<oscSimulationTime>("oscSimulationTime");
	staticObjectFactory.registerType<oscShape>("oscShape");
    staticObjectFactory.registerType<oscSpeed>("oscSpeed");
    staticObjectFactory.registerType<oscSpeedDynamics>("oscSpeedDynamics");
	staticObjectFactory.registerType<oscSetState>("oscSetState");
	staticObjectFactory.registerType<oscSetController>("oscSetController");
	staticObjectFactory.registerType<oscSpline>("oscSpline");
	staticObjectFactory.registerType<oscStandsStill>("oscStandsStill");
	staticObjectFactory.registerType<oscStartCondition>("oscStartCondition");
    staticObjectFactory.registerType<oscStartConditionGroup>("oscStartConditionGroup");
	staticObjectFactory.registerType<oscStartConditionRef>("oscStartConditionRef");
	staticObjectFactory.registerType<oscStoppingDistance>("oscStoppingDistance");
	staticObjectFactory.registerType<oscStoryboard>("oscStoryboard");
	staticObjectFactory.registerType<oscTest>("oscTest");
    staticObjectFactory.registerType<oscTime>("oscTime");  	
    staticObjectFactory.registerType<oscTimeHeadway>("oscTimeHeadway");
	staticObjectFactory.registerType<oscTimeOfDay>("oscTimeOfDay");
	staticObjectFactory.registerType<oscTimeToCollision>("oscTimeToCollision");
	staticObjectFactory.registerType<oscTrafficJam>("oscTrafficJam");
	staticObjectFactory.registerType<oscTrafficLight>("oscTrafficLight");
	staticObjectFactory.registerType<oscTrafficSink>("oscTrafficSink");
	staticObjectFactory.registerType<oscTrafficSource>("oscTrafficSource");
    staticObjectFactory.registerType<oscUserData>("oscUserData");
	staticObjectFactory.registerType<oscUserDefined>("oscUserDefined");
	staticObjectFactory.registerType<oscUserDefinedAction>("oscUserDefinedAction");
	staticObjectFactory.registerType<oscUserDefinedCommand>("oscUserDefinedCommand");
	staticObjectFactory.registerType<oscUserScript>("oscUserScript");
	staticObjectFactory.registerType<oscVehicleAxle>("oscVehicleAxle");
    staticObjectFactory.registerType<oscVehicleCatalog>("oscVehicleCatalog");
	staticObjectFactory.registerType<oscVehicle>("oscVehicle");
	staticObjectFactory.registerType<oscVehicleRef>("oscVehicleRef");
	staticObjectFactory.registerType<oscVelocity>("oscVelocity");
	staticObjectFactory.registerType<oscVisibility>("oscVisibility");
	staticObjectFactory.registerType<oscWaypoint>("oscWaypoint");
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
