/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

// sort alphabetically
#include <oscAbsolute.h>
#include <oscAction.h>
#include <oscAutonomous.h>
#include <oscBehavior.h>
#include <oscBody.h>
#include <oscBoundingBox.h>
#include <oscCatalog.h>
#include <oscCenter.h>
#include <oscColor.h>
#include <oscCondition.h>
#include <oscDate.h>
#include <oscDimension.h>
#include <oscDimensions.h>
#include <oscDirectory.h>
#include <oscDriverCatalog.h>
#include <oscEntityCatalog.h>
#include <oscEnvironment.h>
#include <oscFactories.h>
#include <oscFile.h>
#include <oscFilter.h>
#include <oscFog.h>
#include <oscHeader.h>
#include <oscIntensity.h>
#include <oscLaneChange.h>
#include <oscLaneCoord.h>
#include <oscLaneDynamics.h>
#include <oscLight.h>
#include <oscManeuverCatalog.h>
#include <oscMiscObjectCatalog.h>
#include <oscNamedObject.h>
#include <oscNameId.h>
#include <oscObjectBase.h>
#include <oscObserver.h>
#include <oscObserverCatalog.h>
#include <oscOrientation.h>
#include <oscParameter.h>
#include <oscParameters.h>
#include <oscPedestrianCatalog.h>
#include <oscPositionLane.h>
#include <oscPositionRoad.h>
#include <oscPositionRoute.h>
#include <oscPositionWorld.h>
#include <oscPositionXyz.h>
#include <oscPrecipitation.h>
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
#include <oscSpeed.h>
#include <oscSpeedDynamics.h>
#include <oscStartConditionGroup.h>
#include <oscTime.h>
#include <oscTimeOfDay.h>
#include <oscUserData.h>
#include <oscVariables.h>
#include <oscVehicleCatalog.h>
#include <oscWeather.h>
#include <oscFrustum.h>


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
    staticObjectFactory.registerType<oscAction>("oscAction");
    staticObjectFactory.registerType<oscAutonomous>("oscAutonomous");
    staticObjectFactory.registerType<oscBehavior>("oscBehavior");
    staticObjectFactory.registerType<oscBody>("oscBody");
    staticObjectFactory.registerType<oscBoundingBox>("oscBoundingBox");
    staticObjectFactory.registerType<oscCatalog>("oscCatalog");
    staticObjectFactory.registerType<oscCenter>("oscCenter");
    staticObjectFactory.registerType<oscColor>("oscColor");
    staticObjectFactory.registerType<oscCondition>("oscCondition");
    staticObjectFactory.registerType<oscDate>("oscDate");
    staticObjectFactory.registerType<oscDimension>("oscDimension");
    staticObjectFactory.registerType<oscDimensions>("oscDimensions");
    staticObjectFactory.registerType<oscDirectory>("oscDirectory");
    staticObjectFactory.registerType<oscDriverCatalog>("oscDriverCatalog");
    staticObjectFactory.registerType<oscEntityCatalog>("oscEntityCatalog");
    staticObjectFactory.registerType<oscEnvironment>("oscEnvironment");
    staticObjectFactory.registerType<oscFile>("oscFile");
	staticObjectFactory.registerType<oscFile>("oscFilter");
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
    staticObjectFactory.registerType<oscNamedObject>("oscNamedObject");
    staticObjectFactory.registerType<oscNameId>("oscNameId");
	staticObjectFactory.registerType<oscObserver>("oscObserver");
    staticObjectFactory.registerType<oscObserverCatalog>("oscObserverCatalog");
    staticObjectFactory.registerType<oscOrientation>("oscOrientation");
    staticObjectFactory.registerType<oscParameter>("oscParameter");
    staticObjectFactory.registerType<oscParameters>("oscParameters");
    staticObjectFactory.registerType<oscPedestrianCatalog>("oscPedestrianCatalog");
    staticObjectFactory.registerType<oscPositionLane>("oscPositionLane");
    staticObjectFactory.registerType<oscPositionRoad>("oscPositionRoad");
    staticObjectFactory.registerType<oscPositionRoute>("oscPositionRoute");
    staticObjectFactory.registerType<oscPositionWorld>("oscPositionWorld");
    staticObjectFactory.registerType<oscPositionXyz>("oscPositionXyz");
    staticObjectFactory.registerType<oscPrecipitation>("oscPrecipitation");
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
    staticObjectFactory.registerType<oscSpeed>("oscSpeed");
    staticObjectFactory.registerType<oscSpeedDynamics>("oscSpeedDynamics");
    staticObjectFactory.registerType<oscStartConditionGroup>("oscStartConditionGroup");
    staticObjectFactory.registerType<oscTime>("oscTime");
    staticObjectFactory.registerType<oscTimeOfDay>("oscTimeOfDay");
    staticObjectFactory.registerType<oscUserData>("oscUserData");
    staticObjectFactory.registerType<oscVehicleCatalog>("oscVehicleCatalog");
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
