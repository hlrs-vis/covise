/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

// sort alphabetically
#include "oscAbsoluteTypeA.h"
#include "oscAbsoluteTypeB.h"
#include "oscAction.h"
#include "oscActions.h"
#include "oscAdditionalAxles.h"
#include "oscAerodynamics.h"
#include "oscAfterManeuvers.h"
#include "oscAutonomous.h"
#include "oscAxles.h"
#include "oscBehavior.h"
#include "oscBody.h"
#include "oscBoundingBox.h"
#include "oscCancelConditionsGroupsTypeA.h"
#include "oscCancelConditionsGroupTypeA.h"
#include "oscCancelConditionsTypeA.h"
#include "oscCancelConditionsTypeB.h"
#include "oscCatalog.h"
#include "oscObjectCatalog.h"
#include "oscCatalogReferenceTypeA.h"
#include "oscCatalogs.h"
#include "oscCenter.h"
#include "oscCharacterAppearance.h"
#include "oscCharacterGesture.h"
#include "oscCharacterMotion.h"
#include "oscChoiceController.h"
#include "oscChoiceObject.h"
#include "oscClothoid.h"
#include "oscCog.h"
#include "oscCollision.h"
#include "oscColor.h"
#include "oscCommand.h"
#include "oscConditionBase.h"
#include "oscConditionChoiceBase.h"
#include "oscConditionChoiceTypeA.h"
#include "oscConditionTypeA.h"
#include "oscConditionTypeB.h"
#include "oscContinuation.h"
#include "oscControlPoint.h"
#include "oscCoord.h"
#include "oscCurrentPosition.h"
#include "oscDate.h"
#include "oscDimensionTypeA.h"
#include "oscDimensionTypeB.h"
#include "oscDirectory.h"
#include "oscDistance.h"
#include "oscDistanceLateral.h"
#include "oscDistanceLongitudinal.h"
#include "oscDriver.h"
#include "oscEndConditionsGroupsTypeA.h"
#include "oscEndConditionsGroupTypeA.h"
#include "oscEndConditionsTypeA.h"
#include "oscEndConditionsTypeB.h"
#include "oscEndOfRoad.h"
#include "oscEngine.h"
#include "oscEntities.h"
#include "oscEntity.h"
#include "oscEntityAdd.h"
#include "oscEntityDelete.h"
#include "oscEnvironment.h"
#include "oscEnvironmentReference.h"
#include "oscEvent.h"
#include "oscEvents.h"
#include "oscEyepoint.h"
#include "oscEyepoints.h"
#include "oscFactories.h"
#include "oscFeature.h"
#include "oscFeatures.h"
#include "oscFile.h"
#include "oscFileHeader.h"
#include "oscFilter.h"
#include "oscFilters.h"
#include "oscFog.h"
#include "oscFollowRoute.h"
#include "oscFrustum.h"
#include "oscGearbox.h"
#include "oscGeneral.h"
#include "oscInitDynamics.h"
#include "oscInitState.h"
#include "oscIntensity.h"
#include "oscLaneChange.h"
#include "oscLaneCoord.h"
#include "oscLaneDynamics.h"
#include "oscLaneOffset.h"
#include "oscLaneOffsetDynamics.h"
#include "oscLateral.h"
#include "oscLight.h"
#include "oscLighting.h"
#include "oscLights.h"
#include "oscLongitudinal.h"
#include "oscManeuverGroupsTypeAB.h"
#include "oscManeuverGroupTypeAB.h"
#include "oscManeuversTypeB.h"
#include "oscManeuversTypeC.h"
#include "oscManeuverTypeA.h"
#include "oscManeuverTypeB.h"
#include "oscManeuverTypeC.h"
#include "oscMirror.h"
#include "oscMirrors.h"
#include "oscMiscObject.h"
#include "oscNamePriority.h"
#include "oscNameRefId.h"
#include "oscNameRefIdUserData.h"
#include "oscNameUserData.h"
#include "oscNone.h"
#include "oscNotify.h"
#include "oscNumericCondition.h"
#include "oscObject.h"
#include "oscObjectBase.h"
#include "oscObjects.h"
#include "oscObserverId.h"
#include "oscObserverTypeA.h"
#include "oscObserverTypeB.h"
#include "oscOffroad.h"
#include "oscOrientation.h"
#include "oscOrientationOptional.h"
#include "oscParameterListTypeA.h"
#include "oscParameterListTypeB.h"
#include "oscParameterTypeA.h"
#include "oscParameterTypeB.h"
#include "oscPartnerObject.h"
#include "oscPartnerType.h"
#include "oscPedestrian.h"
#include "oscPedestrianController.h"
#include "oscPerformance.h"
#include "oscPolyline.h"
#include "oscPosition.h"
#include "oscPositionLane.h"
#include "oscPositionRoad.h"
#include "oscPositionRoute.h"
#include "oscPositionWorld.h"
#include "oscPositionXyz.h"
#include "oscPrecipitation.h"
#include "oscPriority.h"
#include "oscReachPosition.h"
#include "oscRefActorsTypeA.h"
#include "oscRefActorsTypeB.h"
#include "oscRefActorTypeA.h"
#include "oscRefActorTypeB.h"
#include "oscReferenceHandling.h"
#include "oscRelativePositionLane.h"
#include "oscRelativePositionRoad.h"
#include "oscRelativePositionWorld.h"
#include "oscRelativeTypeA.h"
#include "oscRelativeTypeB.h"
#include "oscRelativeTypeC.h"
#include "oscRoadCondition.h"
#include "oscRoadConditions.h"
#include "oscRoadConditionsGroup.h"
#include "oscRoadCoord.h"
#include "oscRoadNetwork.h"
#include "oscRoute.h"
#include "oscRouting.h"
#include "oscScenarioEnd.h"
#include "oscSetController.h"
#include "oscSetState.h"
#include "oscShape.h"
#include "oscSimulationTime.h"
#include "oscSpeed.h"
#include "oscSpeedDynamics.h"
#include "oscSpline.h"
#include "oscStandsStill.h"
#include "oscStartConditionsGroupsTypeA.h"
#include "oscStartConditionsGroupsTypeC.h"
#include "oscStartConditionsGroupTypeA.h"
#include "oscStartConditionsGroupTypeC.h"
#include "oscStartConditionsTypeA.h"
#include "oscStartConditionsTypeB.h"
#include "oscStartConditionsTypeC.h"
#include "oscStartConditionTypeC.h"
#include "oscStoppingDistance.h"
#include "oscStoryboard.h"
#include "oscTest.h"
#include "oscTime.h"
#include "oscTimeHeadway.h"
#include "oscTimeOfDay.h"
#include "oscTimeToCollision.h"
#include "oscTiming.h"
#include "oscTrafficJam.h"
#include "oscTrafficLight.h"
#include "oscTrafficSink.h"
#include "oscTrafficSource.h"
#include "oscUserData.h"
#include "oscUserDataList.h"
#include "oscUserDefined.h"
#include "oscUserDefinedAction.h"
#include "oscUserDefinedCommand.h"
#include "oscUserScript.h"
#include "oscVariables.h"
#include "oscVehicle.h"
#include "oscVehicleAxle.h"
#include "oscVisibility.h"
#include "oscVisualRange.h"
#include "oscWaypoint.h"
#include "oscWaypoints.h"
#include "oscWeather.h"


using namespace OpenScenario;


oscFactory<oscObjectBase, std::string> staticObjectFactory;
oscFactory<oscMemberValue, oscMemberValue::MemberTypes> staticValueFactory;

oscFactories* oscFactories::inst = NULL;



/*****
 * constructor
 *****/

oscFactories::oscFactories() :
        // set default factories
        objectFactory(&staticObjectFactory),
        valueFactory(&staticValueFactory)
{
    // register all built-in types
    staticValueFactory.registerType<oscStringValue>(oscMemberValue::STRING);
    staticValueFactory.registerType<oscIntValue>(oscMemberValue::INT);
    staticValueFactory.registerType<oscUIntValue>(oscMemberValue::UINT);
    staticValueFactory.registerType<oscShortValue>(oscMemberValue::SHORT);
    staticValueFactory.registerType<oscUShortValue>(oscMemberValue::USHORT);
    staticValueFactory.registerType<oscDoubleValue>(oscMemberValue::DOUBLE);
    staticValueFactory.registerType<oscBoolValue>(oscMemberValue::BOOL);
    staticValueFactory.registerType<oscFloatValue>(oscMemberValue::FLOAT);
    staticValueFactory.registerType<oscEnumValue>(oscMemberValue::ENUM);
    
    //register all object types
    // sort alphabetically
    staticObjectFactory.registerType<oscAbsoluteTypeA>("oscAbsoluteTypeA");
    staticObjectFactory.registerType<oscAbsoluteTypeB>("oscAbsoluteTypeB");
    staticObjectFactory.registerType<oscAction>("oscAction");
    staticObjectFactory.registerType<oscActions>("oscActions");
    staticObjectFactory.registerType<oscAdditionalAxles>("oscAdditionalAxles");
    staticObjectFactory.registerType<oscAerodynamics>("oscAerodynamics");
    staticObjectFactory.registerType<oscAfterManeuvers>("oscAfterManeuvers");
    staticObjectFactory.registerType<oscAutonomous>("oscAutonomous");
    staticObjectFactory.registerType<oscAxles>("oscAxles");
    staticObjectFactory.registerType<oscBehavior>("oscBehavior");
    staticObjectFactory.registerType<oscBody>("oscBody");
    staticObjectFactory.registerType<oscBoundingBox>("oscBoundingBox");
    staticObjectFactory.registerType<oscCancelConditionsGroupsTypeA>("oscCancelConditionsGroupsTypeA");
    staticObjectFactory.registerType<oscCancelConditionsGroupTypeA>("oscCancelConditionsGroupTypeA");
    staticObjectFactory.registerType<oscCancelConditionsTypeA>("oscCancelConditionsTypeA");
    staticObjectFactory.registerType<oscCancelConditionsTypeB>("oscCancelConditionsTypeB");
    staticObjectFactory.registerType<oscCatalog>("oscCatalog");
    staticObjectFactory.registerType<oscObjectCatalog>("oscObjectCatalog");
    staticObjectFactory.registerType<oscCatalogReferenceTypeA>("oscCatalogReferenceTypeA");
    staticObjectFactory.registerType<oscCatalogs>("oscCatalogs");
    staticObjectFactory.registerType<oscCenter>("oscCenter");
    staticObjectFactory.registerType<oscCharacterAppearance>("oscCharacterAppearance");
    staticObjectFactory.registerType<oscCharacterGesture>("oscCharacterGesture");
    staticObjectFactory.registerType<oscCharacterMotion>("oscCharacterMotion");
    staticObjectFactory.registerType<oscChoiceController>("oscChoiceController");
    staticObjectFactory.registerType<oscChoiceObject>("oscChoiceObject");
    staticObjectFactory.registerType<oscClothoid>("oscClothoid");
    staticObjectFactory.registerType<oscCog>("oscCog");
    staticObjectFactory.registerType<oscCollision>("oscCollision");
    staticObjectFactory.registerType<oscColor>("oscColor");
    staticObjectFactory.registerType<oscCommand>("oscCommand");
    staticObjectFactory.registerType<oscConditionBase>("oscConditionBase");
    staticObjectFactory.registerType<oscConditionChoiceBase>("oscConditionChoiceBase");
    staticObjectFactory.registerType<oscConditionChoiceTypeA>("oscConditionChoiceTypeA");
    staticObjectFactory.registerType<oscConditionTypeA>("oscConditionTypeA");
    staticObjectFactory.registerType<oscConditionTypeB>("oscConditionTypeB");
    staticObjectFactory.registerType<oscContinuation>("oscContinuation");
    staticObjectFactory.registerType<oscControlPoint>("oscControlPoint");
    staticObjectFactory.registerType<oscCoord>("oscCoord");
    staticObjectFactory.registerType<oscCurrentPosition>("oscCurrentPosition");
    staticObjectFactory.registerType<oscDate>("oscDate");
    staticObjectFactory.registerType<oscDimensionTypeA>("oscDimensionTypeA");
    staticObjectFactory.registerType<oscDimensionTypeB>("oscDimensionTypeB");
    staticObjectFactory.registerType<oscDirectory>("oscDirectory");
    staticObjectFactory.registerType<oscDistance>("oscDistance");
    staticObjectFactory.registerType<oscDistanceLateral>("oscDistanceLateral");
    staticObjectFactory.registerType<oscDistanceLongitudinal>("oscDistanceLongitudinal");
    staticObjectFactory.registerType<oscDriver>("oscDriver");
    staticObjectFactory.registerType<oscEndConditionsGroupsTypeA>("oscEndConditionsGroupsTypeA");
    staticObjectFactory.registerType<oscEndConditionsGroupTypeA>("oscEndConditionsGroupTypeA");
    staticObjectFactory.registerType<oscEndConditionsTypeA>("oscEndConditionsTypeA");
    staticObjectFactory.registerType<oscEndConditionsTypeB>("oscEndConditionsTypeB");
    staticObjectFactory.registerType<oscEndOfRoad>("oscEndOfRoad");
    staticObjectFactory.registerType<oscEngine>("oscEngine");
    staticObjectFactory.registerType<oscEntities>("oscEntities");
    staticObjectFactory.registerType<oscEntity>("oscEntity");
    staticObjectFactory.registerType<oscEntityAdd>("oscEntityAdd");
    staticObjectFactory.registerType<oscEntityDelete>("oscEntityDelete");
    staticObjectFactory.registerType<oscEnvironment>("oscEnvironment");
    staticObjectFactory.registerType<oscEnvironmentReference>("oscEnvironmentReference");
    staticObjectFactory.registerType<oscEvent>("oscEvent");
    staticObjectFactory.registerType<oscEvents>("oscEvents");
    staticObjectFactory.registerType<oscEyepoint>("oscEyepoint");
    staticObjectFactory.registerType<oscEyepoints>("oscEyepoints");
    staticObjectFactory.registerType<oscFeature>("oscFeature");
    staticObjectFactory.registerType<oscFeatures>("oscFeatures");
    staticObjectFactory.registerType<oscFile>("oscFile");
    staticObjectFactory.registerType<oscFileHeader>("oscFileHeader");
    staticObjectFactory.registerType<oscFilter>("oscFilter");
    staticObjectFactory.registerType<oscFilters>("oscFilters");
    staticObjectFactory.registerType<oscFog>("oscFog");
    staticObjectFactory.registerType<oscFollowRoute>("oscFollowRoute");
    staticObjectFactory.registerType<oscFrustum>("oscFrustum");
    staticObjectFactory.registerType<oscGearbox>("oscGearbox");
    staticObjectFactory.registerType<oscGeneral>("oscGeneral");
    staticObjectFactory.registerType<oscInitDynamics>("oscInitDynamics");
    staticObjectFactory.registerType<oscInitState>("oscInitState");
    staticObjectFactory.registerType<oscIntensity>("oscIntensity");
    staticObjectFactory.registerType<oscLaneChange>("oscLaneChange");
    staticObjectFactory.registerType<oscLaneCoord>("oscLaneCoord");
    staticObjectFactory.registerType<oscLaneDynamics>("oscLaneDynamics");
    staticObjectFactory.registerType<oscLaneOffset>("oscLaneOffset");
    staticObjectFactory.registerType<oscLaneOffsetDynamics>("oscLaneOffsetDynamics");
    staticObjectFactory.registerType<oscLateral>("oscLateral");
    staticObjectFactory.registerType<oscLight>("oscLight");
    staticObjectFactory.registerType<oscLighting>("oscLighting");
    staticObjectFactory.registerType<oscLights>("oscLights");
    staticObjectFactory.registerType<oscLongitudinal>("oscLongitudinal");
    staticObjectFactory.registerType<oscManeuverGroupsTypeAB>("oscManeuverGroupsTypeAB");
    staticObjectFactory.registerType<oscManeuverGroupTypeAB>("oscManeuverGroupTypeAB");
    staticObjectFactory.registerType<oscManeuversTypeB>("oscManeuversTypeB");
    staticObjectFactory.registerType<oscManeuversTypeC>("oscManeuversTypeC");
    staticObjectFactory.registerType<oscManeuverTypeA>("oscManeuverTypeA");
    staticObjectFactory.registerType<oscManeuverTypeB>("oscManeuverTypeB");
    staticObjectFactory.registerType<oscManeuverTypeC>("oscManeuverTypeC");
    staticObjectFactory.registerType<oscMirror>("oscMirror");
    staticObjectFactory.registerType<oscMirrors>("oscMirrors");
    staticObjectFactory.registerType<oscMiscObject>("oscMiscObject");
    staticObjectFactory.registerType<oscNamePriority>("oscNamePriority");
    staticObjectFactory.registerType<oscNameRefId>("oscNameRefId");
    staticObjectFactory.registerType<oscNameRefIdUserData>("oscNameRefIdUserData");
    staticObjectFactory.registerType<oscNameUserData>("oscNameUserData");
    staticObjectFactory.registerType<oscNone>("oscNone");
    staticObjectFactory.registerType<oscNotify>("oscNotify");
    staticObjectFactory.registerType<oscNumericCondition>("oscNumericCondition");
    staticObjectFactory.registerType<oscObject>("oscObject");
    staticObjectFactory.registerType<oscObjects>("oscObjects");
    staticObjectFactory.registerType<oscObserverId>("oscObserverId");
    staticObjectFactory.registerType<oscObserverTypeA>("oscObserverTypeA");
    staticObjectFactory.registerType<oscObserverTypeB>("oscObserverTypeB");
    staticObjectFactory.registerType<oscOffroad>("oscOffroad");
    staticObjectFactory.registerType<oscOrientation>("oscOrientation");
    staticObjectFactory.registerType<oscOrientationOptional>("oscOrientationOptional");
    staticObjectFactory.registerType<oscParameterListTypeA>("oscParameterListTypeA");
    staticObjectFactory.registerType<oscParameterListTypeB>("oscParameterListTypeB");
    staticObjectFactory.registerType<oscParameterTypeA>("oscParameterTypeA");
    staticObjectFactory.registerType<oscParameterTypeB>("oscParameterTypeB");
    staticObjectFactory.registerType<oscPartnerObject>("oscPartnerObject");
    staticObjectFactory.registerType<oscPartnerType>("oscPartnerType");
    staticObjectFactory.registerType<oscPedestrian>("oscPedestrian");
    staticObjectFactory.registerType<oscPedestrianController>("oscPedestrianController");
    staticObjectFactory.registerType<oscPerformance>("oscPerformance");
    staticObjectFactory.registerType<oscPolyline>("oscPolyline");
    staticObjectFactory.registerType<oscPosition>("oscPosition");
    staticObjectFactory.registerType<oscPositionLane>("oscPositionLane");
    staticObjectFactory.registerType<oscPositionRoad>("oscPositionRoad");
    staticObjectFactory.registerType<oscPositionRoute>("oscPositionRoute");
    staticObjectFactory.registerType<oscPositionWorld>("oscPositionWorld");
    staticObjectFactory.registerType<oscPositionXyz>("oscPositionXyz");
    staticObjectFactory.registerType<oscPrecipitation>("oscPrecipitation");
    staticObjectFactory.registerType<oscPriority>("oscPriority");
    staticObjectFactory.registerType<oscReachPosition>("oscReachPosition");
    staticObjectFactory.registerType<oscRefActorsTypeA>("oscRefActorsTypeA");
    staticObjectFactory.registerType<oscRefActorsTypeB>("oscRefActorsTypeB");
    staticObjectFactory.registerType<oscRefActorTypeA>("oscRefActorTypeA");
    staticObjectFactory.registerType<oscRefActorTypeB>("oscRefActorTypeB");
    staticObjectFactory.registerType<oscReferenceHandling>("oscReferenceHandling");
    staticObjectFactory.registerType<oscRelativePositionLane>("oscRelativePositionLane");
    staticObjectFactory.registerType<oscRelativePositionRoad>("oscRelativePositionRoad");
    staticObjectFactory.registerType<oscRelativePositionWorld>("oscRelativePositionWorld");
    staticObjectFactory.registerType<oscRelativeTypeA>("oscRelativeTypeA");
    staticObjectFactory.registerType<oscRelativeTypeB>("oscRelativeTypeB");
    staticObjectFactory.registerType<oscRelativeTypeC>("oscRelativeTypeC");
    staticObjectFactory.registerType<oscRoadCondition>("oscRoadCondition");
    staticObjectFactory.registerType<oscRoadConditions>("oscRoadConditions");
    staticObjectFactory.registerType<oscRoadConditionsGroup>("oscRoadConditionsGroup");
    staticObjectFactory.registerType<oscRoadCoord>("oscRoadCoord");
    staticObjectFactory.registerType<oscRoadNetwork>("oscRoadNetwork");
    staticObjectFactory.registerType<oscRoute>("oscRoute");
    staticObjectFactory.registerType<oscRouting>("oscRouting");
    staticObjectFactory.registerType<oscScenarioEnd>("oscScenarioEnd");
    staticObjectFactory.registerType<oscSetController>("oscSetController");
    staticObjectFactory.registerType<oscSetState>("oscSetState");
    staticObjectFactory.registerType<oscShape>("oscShape");
    staticObjectFactory.registerType<oscSimulationTime>("oscSimulationTime");
    staticObjectFactory.registerType<oscSpeed>("oscSpeed");
    staticObjectFactory.registerType<oscSpeedDynamics>("oscSpeedDynamics");
    staticObjectFactory.registerType<oscSpline>("oscSpline");
    staticObjectFactory.registerType<oscStandsStill>("oscStandsStill");
    staticObjectFactory.registerType<oscStartConditionsGroupsTypeA>("oscStartConditionsGroupsTypeA");
    staticObjectFactory.registerType<oscStartConditionsGroupsTypeC>("oscStartConditionsGroupsTypeC");
    staticObjectFactory.registerType<oscStartConditionsGroupTypeA>("oscStartConditionsGroupTypeA");
    staticObjectFactory.registerType<oscStartConditionsGroupTypeC>("oscStartConditionsGroupTypeC");
    staticObjectFactory.registerType<oscStartConditionsTypeA>("oscStartConditionsTypeA");
    staticObjectFactory.registerType<oscStartConditionsTypeB>("oscStartConditionsTypeB");
    staticObjectFactory.registerType<oscStartConditionsTypeC>("oscStartConditionsTypeC");
    staticObjectFactory.registerType<oscStartConditionTypeC>("oscStartConditionTypeC");
    staticObjectFactory.registerType<oscStoppingDistance>("oscStoppingDistance");
    staticObjectFactory.registerType<oscStoryboard>("oscStoryboard");
    staticObjectFactory.registerType<oscTest>("oscTest");
    staticObjectFactory.registerType<oscTime>("oscTime");
    staticObjectFactory.registerType<oscTimeHeadway>("oscTimeHeadway");
    staticObjectFactory.registerType<oscTimeOfDay>("oscTimeOfDay");
    staticObjectFactory.registerType<oscTimeToCollision>("oscTimeToCollision");
    staticObjectFactory.registerType<oscTiming>("oscTiming");
    staticObjectFactory.registerType<oscTrafficJam>("oscTrafficJam");
    staticObjectFactory.registerType<oscTrafficLight>("oscTrafficLight");
    staticObjectFactory.registerType<oscTrafficSink>("oscTrafficSink");
    staticObjectFactory.registerType<oscTrafficSource>("oscTrafficSource");
    staticObjectFactory.registerType<oscUserData>("oscUserData");
    staticObjectFactory.registerType<oscUserDataList>("oscUserDataList");
    staticObjectFactory.registerType<oscUserDefined>("oscUserDefined");
    staticObjectFactory.registerType<oscUserDefinedAction>("oscUserDefinedAction");
    staticObjectFactory.registerType<oscUserDefinedCommand>("oscUserDefinedCommand");
    staticObjectFactory.registerType<oscUserScript>("oscUserScript");
    staticObjectFactory.registerType<oscVehicle>("oscVehicle");
    staticObjectFactory.registerType<oscVehicleAxle>("oscVehicleAxle");
    staticObjectFactory.registerType<oscVisibility>("oscVisibility");
    staticObjectFactory.registerType<oscVisualRange>("oscVisualRange");
    staticObjectFactory.registerType<oscWaypoint>("oscWaypoint");
    staticObjectFactory.registerType<oscWaypoints>("oscWaypoints");
    staticObjectFactory.registerType<oscWeather>("oscWeather");
//    staticObjectFactory.registerType<>("");
}



/*****
 * destructor
 *****/

oscFactories::~oscFactories()
{

}



/*****
 * public functions
 *****/

oscFactories *oscFactories::instance()
{
    if(inst == NULL)
    {
        inst = new oscFactories();
    }

    return inst;
}


void oscFactories::setObjectFactory(oscFactory<oscObjectBase, std::string> *f)
{
    objectFactory = f;
    if(objectFactory == NULL)
        objectFactory = &staticObjectFactory;
}

void oscFactories::setValueFactory(oscFactory<oscMemberValue, oscMemberValue::MemberTypes> *f)
{
    valueFactory = f;
    if(valueFactory == NULL)
        valueFactory = &staticValueFactory;
}
