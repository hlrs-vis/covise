/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscFactories.h>
#include <oscHeader.h>
#include <oscFile.h>
#include <oscDirectory.h>
#include <oscEnvironment.h>
#include <oscWeather.h>
#include <oscTimeOfDay.h>
#include <oscRoadNetwork.h>
#include <oscVariables.h>
#include <oscObjectBase.h>


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
    
    // sorted alphabetically
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
    staticObjectFactory.registerType<oscFog>("oscFog");
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
