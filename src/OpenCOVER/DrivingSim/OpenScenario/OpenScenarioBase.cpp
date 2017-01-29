/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "OpenScenarioBase.h"
#include "oscVariables.h"

#include <iostream>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/validators/common/Grammar.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/MemBufFormatTarget.hpp>
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/util/XercesVersion.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>


using namespace OpenScenario;


/*****
 * constructor
 *****/

OpenScenarioBase::OpenScenarioBase() :
        oscObjectBase(),
        xmlDoc(NULL),
        m_validate(true),
        m_fullReadCatalogs(false)
{
    oscFactories::instance();
    OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader", 0);
    OSC_OBJECT_ADD_MEMBER(Catalogs, "oscCatalogs", 0);
    OSC_OBJECT_ADD_MEMBER(RoadNetwork, "oscRoadNetwork", 0);
    OSC_OBJECT_ADD_MEMBER(Entities, "oscEntities", 0);
    OSC_OBJECT_ADD_MEMBER(Storyboard, "oscStoryboard", 0);

    base = this;

    ownMember = new oscMember();
    ownMember->setName("OpenSCENARIO");
    ownMember->setTypeName("OpenScenarioBase");
    ownMember->setValue(this);

    //in order to work with the Xerces-C++ parser, the XML subsystem must be initialized first
    //every call of XMLPlatformUtils::Initialize() must have a matching call of XMLPlatformUtils::Terminate() (see destructor)
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cerr << "Error during xerces initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
    }

    //parser and error handler have to be initialized _after_ xercesc::XMLPlatformUtils::Initialize()
    //can't be done in member initializer list
    parser = new xercesc::XercesDOMParser();
    parserErrorHandler = new ParserErrorHandler();

    //generic settings for parser
    //
    parser->setErrorHandler(parserErrorHandler);
    parser->setExitOnFirstFatalError(true);
    //namespaces needed for XInclude and validation
    parser->setDoNamespaces(true);
    //settings for validation
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);
    parser->setValidationConstraintFatal(true);
    parser->cacheGrammarFromParse(true);
}



/*****
 * initialization static variables
 *****/

OpenScenarioBase::FileTypeXsdFileNameMap initFuncFileTypeToXsd()
{
    //set the XSD Schema file name for possible file types
    OpenScenarioBase::FileTypeXsdFileNameMap fileTypeToXsd;
//	fileTypeToXsd.emplace("", bf::path("OpenScenario_XML-Schema_.xsd"));
    fileTypeToXsd.emplace("OpenSCENARIO", bf::path("OpenSCENARIO_Draft_F.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_DriverCatalog", bf::path("OpenSCENARIO_DriverCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_EnvironmentCatalog", bf::path("OpenSCENARIO_EnvironmentCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_ManeuverCatalog", bf::path("OpenSCENARIO_ManeuverCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_MiscObjectCatalog", bf::path("OpenSCENARIO_MiscObjectCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_PedestrianCatalog", bf::path("OpenSCENARIO_PedestrianCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_PedestrianControllerCatalog", bf::path("OpenSCENARIO_PedestrianControllerCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_RouteCatalog", bf::path("OpenSCENARIO_RouteCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_TrajectoryCatalog", bf::path("OpenSCENARIO_TrajectoryCatalog.xsd"));
	fileTypeToXsd.emplace("OpenSCENARIO_VehicleCatalog", bf::path("OpenSCENARIO_VehicleCatalog.xsd"));

/*	fileTypeToXsd.emplace("oscAbsoluteLaneOffsetTypeB", bf::path("OpenScenario_XML-Schema_absoluteLaneOffsetTypeB.xsd"));
	fileTypeToXsd.emplace("oscAbsoluteTypeA", bf::path("OpenScenario_XML-Schema_absoluteTypeA.xsd"));
	fileTypeToXsd.emplace("oscAbsoluteTypeB", bf::path("OpenScenario_XML-Schema_absoluteTypeB.xsd"));
	fileTypeToXsd.emplace("oscAction", bf::path("OpenScenario_XML-Schema_action.xsd"));
	fileTypeToXsd.emplace("oscAerodynamics", bf::path("OpenScenario_XML-Schema_aerodynamics.xsd"));
	fileTypeToXsd.emplace("oscAfterManeuvers", bf::path("OpenScenario_XML-Schema_afterManeuvers.xsd"));
	fileTypeToXsd.emplace("oscAutonomous", bf::path("OpenScenario_XML-Schema_autonomous.xsd"));
	fileTypeToXsd.emplace("oscAxles", bf::path("OpenScenario_XML-Schema_axles.xsd"));
	fileTypeToXsd.emplace("oscBehavior", bf::path("OpenScenario_XML-Schema_behavior.xsd"));
	fileTypeToXsd.emplace("oscBody", bf::path("OpenScenario_XML-Schema_body.xsd"));
	fileTypeToXsd.emplace("oscBoundingBox", bf::path("OpenScenario_XML-Schema_boundingBox.xsd"));
	fileTypeToXsd.emplace("oscCenter", bf::path("OpenScenario_XML-Schema_center.xsd"));
	fileTypeToXsd.emplace("oscCharacterAppearance", bf::path("OpenScenario_XML-Schema_characterAppearance.xsd"));
	fileTypeToXsd.emplace("oscCharacterGesture", bf::path("OpenScenario_XML-Schema_characterGesture.xsd"));
	fileTypeToXsd.emplace("oscCharacterMotion", bf::path("OpenScenario_XML-Schema_characterMotion.xsd"));
	fileTypeToXsd.emplace("oscChoiceController", bf::path("OpenScenario_XML-Schema_controllerChoice.xsd"));
	fileTypeToXsd.emplace("oscClothoid", bf::path("OpenScenario_XML-Schema_clothoid.xsd"));
	fileTypeToXsd.emplace("oscCog", bf::path("OpenScenario_XML-Schema_cog.xsd"));
	fileTypeToXsd.emplace("oscTimeToCollision", bf::path("OpenScenario_XML-Schema_collision.xsd"));
	fileTypeToXsd.emplace("oscColor", bf::path("OpenScenario_XML-Schema_color.xsd"));
	fileTypeToXsd.emplace("oscCommand", bf::path("OpenScenario_XML-Schema_command.xsd"));
	fileTypeToXsd.emplace("oscConditionBase", bf::path("OpenScenario_XML-Schema_condition.xsd"));
	fileTypeToXsd.emplace("oscConditionChoiceTypeA", bf::path("OpenScenario_XML-Schema_velocity.xsc"));
	fileTypeToXsd.emplace("oscContinuation", bf::path("OpenScenario_XML-Schema_continuation.xsd"));
	fileTypeToXsd.emplace("oscCoord", bf::path("OpenScenario_XML-Schema_coord.xsd"));
	fileTypeToXsd.emplace("oscCurrentPosition", bf::path("OpenScenario_XML-Schema_currentPosition.xsd"));
	fileTypeToXsd.emplace("oscDate", bf::path("OpenScenario_XML-Schema_date.xsd"));
	fileTypeToXsd.emplace("oscDimensionTypeA", bf::path("OpenScenario_XML-Schema_dimensionTypeA.xsd"));
	fileTypeToXsd.emplace("oscDimensionTypeB", bf::path("OpenScenario_XML-Schema_dimensionTypeB.xsd"));
	fileTypeToXsd.emplace("oscDistance", bf::path("OpenScenario_XML-Schema_distance.xsd"));
	fileTypeToXsd.emplace("oscDistanceLateral", bf::path("OpenScenario_XML-Schema_distanceLateral.xsd"));
	fileTypeToXsd.emplace("oscDistanceLongitudinal", bf::path("OpenScenario_XML-Schema_distanceLongitudinal.xsd"));
    fileTypeToXsd.emplace("oscDriver", bf::path("OpenScenario_XML-Schema_driver.xsd"));
	fileTypeToXsd.emplace("dynamics", bf::path("OpenScenario_XML-Schema_dynamics.xsd"));
	fileTypeToXsd.emplace("oscEndOfRoad", bf::path("OpenScenario_XML-Schema_afterManeuvers.xsd"));
	fileTypeToXsd.emplace("oscEngine", bf::path("OpenScenario_XML-Schema_engine.xsd"));	
	fileTypeToXsd.emplace("oscEntity", bf::path("OpenScenario_XML-Schema_entity.xsd"));
	fileTypeToXsd.emplace("oscEntityAdd", bf::path("OpenScenario_XML-Schema_entityAdd.xsd"));
	fileTypeToXsd.emplace("oscEntityDelete", bf::path("OpenScenario_XML-Schema_entityDelete.xsd"));
    fileTypeToXsd.emplace("oscEnvironment", bf::path("OpenScenario_XML-Schema_environment.xsd"));
	fileTypeToXsd.emplace("oscEvent", bf::path("OpenScenario_XML-Schema_event.xsd"));
	fileTypeToXsd.emplace("oscEyepoint", bf::path("OpenScenario_XML-Schema_eyepoint.xsd"));
	fileTypeToXsd.emplace("oscFeature", bf::path("OpenScenario_XML-Schema_feature.xsd"));
	fileTypeToXsd.emplace("oscFile", bf::path("OpenScenario_XML-Schema_file.xsd"));
	fileTypeToXsd.emplace("oscFileHeader", bf::path("OpenScenario_XML-Schema_fileHeader.xsd"));
	fileTypeToXsd.emplace("oscFilter", bf::path("OpenScenario_XML-Schema_filter.xsd"));
	fileTypeToXsd.emplace("oscFog", bf::path("OpenScenario_XML-Schema_fog.xsd"));
	fileTypeToXsd.emplace("oscFollowRoute", bf::path("OpenScenario_XML-Schema_followRoute.xsd"));
	fileTypeToXsd.emplace("oscFrustum", bf::path("OpenScenario_XML-Schema_frustum.xsd"));
	fileTypeToXsd.emplace("oscGearbox", bf::path("OpenScenario_XML-Schema_gearbox.xsd"));
	fileTypeToXsd.emplace("oscGeneral", bf::path("OpenScenario_XML-Schema_general.xsd"));
	fileTypeToXsd.emplace("oscGeometry", bf::path("OpenScenario_XML-Schema_geometry.xsd"));
	fileTypeToXsd.emplace("oscIntensity", bf::path("OpenScenario_XML-Schema_intensity.xsd"));
	fileTypeToXsd.emplace("oscLaneChange", bf::path("OpenScenario_XML-Schema_laneChange.xsd"));
	fileTypeToXsd.emplace("oscLaneCoord", bf::path("OpenScenario_XML-Schema_laneCoord.xsd"));
	fileTypeToXsd.emplace("oscLaneDynamics", bf::path("OpenScenario_XML-Schema_laneDynamics.xsd"));
	fileTypeToXsd.emplace("oscLaneOffset", bf::path("OpenScenario_XML-Schema_laneOffset.xsd"));
	fileTypeToXsd.emplace("oscLaneOffsetDynamics", bf::path("OpenScenario_XML-Schema_laneOffsetDynamics.xsd"));
	fileTypeToXsd.emplace("oscLateral", bf::path("OpenScenario_XML-Schema_lateral.xsd"));
	fileTypeToXsd.emplace("oscLight", bf::path("OpenScenario_XML-Schema_light.xsd"));
	fileTypeToXsd.emplace("oscLights", bf::path("OpenScenario_XML-Schema_lights.xsd"));
	fileTypeToXsd.emplace("oscLongitudinal", bf::path("OpenScenario_XML-Schema_longitudinal.xsd"));
    fileTypeToXsd.emplace("oscManeuver", bf::path("OpenScenario_XML-Schema_maneuver.xsd"));
	fileTypeToXsd.emplace("oscMirror", bf::path("OpenScenario_XML-Schema_mirror.xsd"));	
    fileTypeToXsd.emplace("oscMiscObject", bf::path("OpenScenario_XML-Schema_miscObject.xsd"));
	fileTypeToXsd.emplace("oscNameRefId", bf::path("OpenScenario_XML-Schema_nameRefId.xsd"));
	fileTypeToXsd.emplace("oscNotify", bf::path("OpenScenario_XML-Schema_notify.xsd"));
	fileTypeToXsd.emplace("oscNumericCondition", bf::path("OpenScenario_XML-Schema_numericCondition.xsd"));
	fileTypeToXsd.emplace("oscChoiceObject", bf::path("OpenScenario_XML-Schema_objectChoice.xsd"));
    fileTypeToXsd.emplace("oscObserverTypeA", bf::path("OpenScenario_XML-Schema_observerTypeA.xsd"));
	fileTypeToXsd.emplace("oscObserverTypeB", bf::path("OpenScenario_XML-Schema_observerTypeB.xsd"));	
	fileTypeToXsd.emplace("oscObserverId", bf::path("OpenScenario_XML-Schema_observerId.xsd"));
	fileTypeToXsd.emplace("oscOffroad", bf::path("OpenScenario_XML-Schema_offroad.xsd"));
	fileTypeToXsd.emplace("oscOrientation", bf::path("OpenScenario_XML-Schema_orientation.xsd"));
	fileTypeToXsd.emplace("oscParameterTypeA", bf::path("OpenScenario_XML-Schema_parameterTypeA.xsd"));
	fileTypeToXsd.emplace("oscParameterTypeB", bf::path("OpenScenario_XML-Schema_parameterTypeB.xsd"));
	fileTypeToXsd.emplace("oscPartnerType", bf::path("OpenScenario_XML-Schema_partnerType.xsd"));
	fileTypeToXsd.emplace("oscPartnerObject", bf::path("OpenScenario_XML-Schema_partnerObject.xsd"));
    fileTypeToXsd.emplace("oscPedestrian", bf::path("OpenScenario_XML-Schema_pedestrian.xsd"));
	fileTypeToXsd.emplace("oscPedestrianController", bf::path("OpenScenario_XML-Schema_pedestrianController.xsd"));
	fileTypeToXsd.emplace("oscPerformance", bf::path("OpenScenario_XML-Schema_performance.xsd"));
	fileTypeToXsd.emplace("oscPosition", bf::path("OpenScenario_XML-Schema_position.xsd"));
	fileTypeToXsd.emplace("oscPositionLane", bf::path("OpenScenario_XML-Schema_positionLane.xsd"));
	fileTypeToXsd.emplace("oscPositionRoad", bf::path("OpenScenario_XML-Schema_positionRoad.xsd"));
	fileTypeToXsd.emplace("oscPositionRoute", bf::path("OpenScenario_XML-Schema_positionRoute.xsd"));
	fileTypeToXsd.emplace("oscPositionWorld", bf::path("OpenScenario_XML-Schema_positionWorld.xsd"));
	fileTypeToXsd.emplace("oscPositionXyz", bf::path("OpenScenario_XML-Schema_positionXyz.xsd"));
	fileTypeToXsd.emplace("oscPrecipitation", bf::path("OpenScenario_XML-Schema_precipitation.xsd"));
	fileTypeToXsd.emplace("oscReachPosition", bf::path("OpenScenario_XML-Schema_reachPosition.xsd"));
	fileTypeToXsd.emplace("oscReferenceHandling", bf::path("OpenScenario_XML-Schema_referenceHandling.xsd"));
	fileTypeToXsd.emplace("oscRelativePositionLane", bf::path("OpenScenario_XML-Schema_relativePositionLane.xsd"));	
	fileTypeToXsd.emplace("oscRelativePositionLane", bf::path("OpenScenario_XML-Schema_relativePositionLane.xsd"));
	fileTypeToXsd.emplace("oscRelativePositionRoad", bf::path("OpenScenario_XML-Schema_relativePositionRoad.xsd"));
	fileTypeToXsd.emplace("oscRelativePositionWorld", bf::path("OpenScenario_XML-Schema_relativePositionWorld.xsd"));
	fileTypeToXsd.emplace("oscRelativeSpeedTypeC", bf::path("OpenScenario_XML-Schema_relativeSpeedTypeC.xsd"));
	fileTypeToXsd.emplace("oscRelativeTypeA", bf::path("OpenScenario_XML-Schema_relativeTypeA.xsd"));
	fileTypeToXsd.emplace("oscRelativeTypeB", bf::path("OpenScenario_XML-Schema_relativeTypeB.xsd"));
	fileTypeToXsd.emplace("oscRoadCondition", bf::path("OpenScenario_XML-Schema_roadCondition.xsd"));
	fileTypeToXsd.emplace("oscRoadConditionsGroup", bf::path("OpenScenario_XML-Schema_roadConditionsGroup.xsd"));
	fileTypeToXsd.emplace("oscRoadCoord", bf::path("OpenScenario_XML-Schema_roadCoord.xsd"));
	fileTypeToXsd.emplace("oscRoute", bf::path("OpenScenario_XML-Schema_route.xsd"));
    fileTypeToXsd.emplace("oscRouting", bf::path("OpenScenario_XML-Schema_routing.xsd"));
	fileTypeToXsd.emplace("oscSetController", bf::path("OpenScenario_XML-Schema_setController.xsd"));
	fileTypeToXsd.emplace("oscSetState", bf::path("OpenScenario_XML-Schema_setState.xsd"));
	fileTypeToXsd.emplace("oscShape", bf::path("OpenScenario_XML-Schema_shape.xsd"));
	fileTypeToXsd.emplace("oscSimulationTime", bf::path("OpenScenario_XML-Schema_simulationTime.xsd"));
	fileTypeToXsd.emplace("oscSpeed", bf::path("OpenScenario_XML-Schema_speed.xsd"));
	fileTypeToXsd.emplace("oscSpeedDynamics", bf::path("OpenScenario_XML-Schema_speedDynamics.xsd"));
	fileTypeToXsd.emplace("oscSpline", bf::path("OpenScenario_XML-Schema_spline.xsd"));
	fileTypeToXsd.emplace("oscStandsStill", bf::path("OpenScenario_XML-Schema_standsStill.xsd"));
	fileTypeToXsd.emplace("oscStartConditionsGroupsTypeC", bf::path("OpenScenario_XML-Schema_startConditionGroupsTypeC.xsd"));
	fileTypeToXsd.emplace("oscStartConditionsGroupTypeC", bf::path("OpenScenario_XML-Schema_startConditionGroupTypeC.xsd"));
	fileTypeToXsd.emplace("oscStartConditionTypeC", bf::path("OpenScenario_XML-Schema_startConditionTypeC.xsd"));
	fileTypeToXsd.emplace("oscStoppingDistance", bf::path("OpenScenario_XML-Schema_stoppingDistance.xsd"));
	fileTypeToXsd.emplace("oscTime", bf::path("OpenScenario_XML-Schema_time.xsd"));
	fileTypeToXsd.emplace("oscTimeHeadway", bf::path("OpenScenario_XML-Schema_stoppingDistance.xsd"));
	fileTypeToXsd.emplace("oscTimeToCollision", bf::path("OpenScenario_XML-Schema_stoppingDistance.xsd"));
	fileTypeToXsd.emplace("oscTimeOfDay", bf::path("OpenScenario_XML-Schema_timeOfDay.xsd"));
	fileTypeToXsd.emplace("oscTimeOfDayWithAnimation", bf::path("OpenScenario_XML-Schema_timeOfDayWithAnimation.xsd"));
	fileTypeToXsd.emplace("oscTiming", bf::path("OpenScenario_XML-Schema_timing.xsd"));
	fileTypeToXsd.emplace("oscTrafficJam", bf::path("OpenScenario_XML-Schema_trafficJam.xsd"));
	fileTypeToXsd.emplace("oscTrafficLight", bf::path("OpenScenario_XML-Schema_trafficLight.xsd"));
	fileTypeToXsd.emplace("oscTrafficSink", bf::path("OpenScenario_XML-Schema_trafficSink.xsd"));
	fileTypeToXsd.emplace("oscTrafficSource", bf::path("OpenScenario_XML-Schema_trafficSource.xsd"));
	fileTypeToXsd.emplace("oscUserData", bf::path("OpenScenario_XML-Schema_userData.xsd"));
	fileTypeToXsd.emplace("oscUserDatalist", bf::path("OpenScenario_XML-Schema_userDataList.xsd"));
	fileTypeToXsd.emplace("oscUserDefinedAction", bf::path("OpenScenario_XML-Schema_userDefinedAction.xsd"));
	fileTypeToXsd.emplace("oscUserDefinedCommand", bf::path("OpenScenario_XML-Schema_userDefinedCommand.xsd"));
	fileTypeToXsd.emplace("oscUserScript", bf::path("OpenScenario_XML-Schema_userScript.xsd"));
    fileTypeToXsd.emplace("oscVehicle", bf::path("OpenScenario_XML-Schema_vehicle.xsd"));
	fileTypeToXsd.emplace("oscVehicleAxle", bf::path("OpenScenario_XML-Schema_vehicleAxle.xsd"));
	fileTypeToXsd.emplace("oscVelocity", bf::path("OpenScenario_XML-Schema_velocity.xsd"));
	fileTypeToXsd.emplace("oscVisibility", bf::path("OpenScenario_XML-Schema_visibility.xsd"));
	fileTypeToXsd.emplace("oscVisualRange", bf::path("OpenScenario_XML-Schema_visualRange.xsd"));
	fileTypeToXsd.emplace("oscWaypoint", bf::path("OpenScenario_XML-Schema_waypoint.xsd"));
	fileTypeToXsd.emplace("oscWeather", bf::path("OpenScenario_XML-Schema_weather.xsd")); */

    return fileTypeToXsd;
}

const OpenScenarioBase::FileTypeXsdFileNameMap OpenScenarioBase::s_fileTypeToXsdFileName = initFuncFileTypeToXsd();

OpenScenarioBase::DefaultFileTypeNameMap initDefaultFileTypeMap()
{
    //set the XSD Schema file name for possible file types
    OpenScenarioBase::DefaultFileTypeNameMap defaultFileTypeMap;
	
	defaultFileTypeMap.emplace("oscAction", bf::path("OpenScenario_XML-Default_OSCAction.xosc"));	
	defaultFileTypeMap.emplace("oscDriver", bf::path("OpenScenario_XML-Default_driver.xosc"));
	defaultFileTypeMap.emplace("oscEntity", bf::path("OpenScenario_XML-Default_entity.xosc"));
	defaultFileTypeMap.emplace("oscEnvironment", bf::path("OpenScenario_XML-Default_environment.xosc"));
	defaultFileTypeMap.emplace("oscEvent", bf::path("OpenScenario_XML-Default_OSCEvent.xosc"));	
	defaultFileTypeMap.emplace("oscFilter", bf::path("OpenScenario_XML-Default_OSCFilter.xosc"));
	defaultFileTypeMap.emplace("oscLight", bf::path("OpenScenario_XML-Default_OSCLight.xosc"));
	defaultFileTypeMap.emplace("oscManeuver", bf::path("OpenScenario_XML-Default_maneuver.xosc"));
	defaultFileTypeMap.emplace("oscMiscObject", bf::path("OpenScenario_XML-Default_miscObject.xosc"));	
	defaultFileTypeMap.emplace("oscObserver", bf::path("OpenScenario_XML-Default_observer.xosc"));	
	defaultFileTypeMap.emplace("oscParameter", bf::path("OpenScenario_XML-Default_OSCParameterTypeA.xosc"));
	defaultFileTypeMap.emplace("oscPedestrian", bf::path("OpenScenario_XML-Default_pedestrian.xosc"));
	defaultFileTypeMap.emplace("oscRoadCondition", bf::path("OpenScenario_XML-Default_OSCRoadCondition.xosc"));
	defaultFileTypeMap.emplace("oscRouting", bf::path("OpenScenario_XML-Default_routing.xosc"));
	defaultFileTypeMap.emplace("oscStartCondition", bf::path("OpenScenario_XML-Default_OSCStartCondition.xosc"));defaultFileTypeMap.emplace("oscStartCondition", bf::path("OpenScenario_XML-Default_OSCStartCondition.xosc"));
	defaultFileTypeMap.emplace("oscUserData", bf::path("OpenScenario_XML-Default_OSCUserData.xosc"));
	defaultFileTypeMap.emplace("oscVehicle", bf::path("OpenScenario_XML-Default_vehicle.xosc"));
	defaultFileTypeMap.emplace("oscWaypoint", bf::path("OpenScenario_XML-Default_OSCWaypoint.xosc"));

    return defaultFileTypeMap;
}

const OpenScenarioBase::FileTypeXsdFileNameMap OpenScenarioBase::s_defaultFileTypeMap = initDefaultFileTypeMap();




/*****
 * destructor
 *****/

OpenScenarioBase::~OpenScenarioBase()
{
    delete parserErrorHandler;
    delete parser;
    //match the call of XMLPlatformUtils::Initialize() from constructor
    try
    {
        xercesc::XMLPlatformUtils::Terminate();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during xerces termination! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
    }
}



/*****
 * public functions
 *****/

//
xercesc::DOMDocument *OpenScenarioBase::getDocument() const
{
    return xmlDoc;
}


//
void OpenScenarioBase::addToSrcFileVec(oscSourceFile *src)
{
    srcFileVec.push_back(src);
}

std::vector<oscSourceFile *> OpenScenarioBase::getSrcFileVec() const
{
    return srcFileVec;
}


//
void OpenScenarioBase::setValidation(const bool validate)
{
    m_validate = validate;
}

bool OpenScenarioBase::getValidation() const
{
    return m_validate;
}

void OpenScenarioBase::setFullReadCatalogs(const bool fullReadCatalogs)
{
    m_fullReadCatalogs = fullReadCatalogs;
}

bool OpenScenarioBase::getFullReadCatalogs() const
{
    return m_fullReadCatalogs;
}


//
void OpenScenarioBase::setPathFromCurrentDirToDoc(const bf::path &path)
{
    m_pathFromCurrentDirToDoc = path.parent_path();
}

void OpenScenarioBase::setPathFromCurrentDirToDoc(const std::string &path)
{
    m_pathFromCurrentDirToDoc = bf::path(path).parent_path();
}

bf::path OpenScenarioBase::getPathFromCurrentDirToDoc() const
{
    return m_pathFromCurrentDirToDoc;
}


//
bool OpenScenarioBase::loadFile(const std::string &fileName, const std::string &nodeName, const std::string &fileType)
{
    setPathFromCurrentDirToDoc(fileName);

    xercesc::DOMElement *rootElement = getRootElement(fileName, nodeName, fileType, m_validate);
/*	if(rootElement)
	{
	std::string rootElementName = xercesc::XMLString::transcode(rootElement->getNodeName());

	if (rootElementName != fileType)
	{
		return false;
	}
	} */

    if(rootElement == NULL) // create new file
    {
		if (createSource(fileName, fileType))
		{
			return true;
		}
		return false;
    }
    else
    {
        return parseFromXML(rootElement);
    }
}

bool OpenScenarioBase::saveFile(const std::string &fileName, bool overwrite/* default false */)
{
    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();
    //oscSourceFile for OpenScenarioBase
    oscSourceFile *osbSourceFile;

    //create xmlDocs for every source file
    //
    for (int i = 0; i < srcFileVec.size(); i++)
    {
        std::string srcFileRootElement = srcFileVec[i]->getRootElementNameAsStr();

        //set filename for main xosc file with root element "OpenSCENARIO" to fileName
		bf::path fnPath = srcFileVec[i]->getFileNamePath(fileName);
        if (srcFileRootElement == "OpenSCENARIO") 
        {
			if ((fnPath == "") || (fnPath.parent_path() == srcFileVec[i]->getAbsPathToMainDir()))
			{
				srcFileVec[i]->setSrcFileName(fnPath.filename());
			}
			else
			{
				source = new oscSourceFile();
				source->setNameAndPath(fileName, "OpenSCENARIO", m_pathFromCurrentDirToDoc);
				srcFileVec[i] = source;
			}
			osbSourceFile = srcFileVec[i];
        }
		
        if (!srcFileVec[i]->getXmlDoc())
        {
            xercesc::DOMDocument *xmlSrcDoc = impl->createDocument(0, srcFileVec[i]->getRootElementNameAsXmlCh(), 0);
            srcFileVec[i]->setXmlDoc(xmlSrcDoc);
        }
    }

    //write objects to their own xmlDoc
    //start with the document with root element OpenSCENARIO
    //
    writeToDOM(osbSourceFile->getXmlDoc()->getDocumentElement(), osbSourceFile->getXmlDoc());

    //write xmlDocs to disk
    //
    for (int i = 0; i < srcFileVec.size(); i++)
    {
        //get the file name to write
        bf::path srcFileName = srcFileVec[i]->getSrcFileName();

        //////
        //for testing: generate a new filename
        //
        if (srcFileName != fileName)
        {
            srcFileName += "_out.xml";
        }
        //
        //////

		srcFileVec[i]->writeFileToDisk();
    }

    return true;
}

void OpenScenarioBase::clearDOM()
{
	for (int i = 0; i < srcFileVec.size(); i++)
	{
		srcFileVec[i]->clearXmlDoc();
	}
}


xercesc::MemBufFormatTarget *OpenScenarioBase::writeFileToMemory(xercesc::DOMDocument *xmlDocToWrite)
{
    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();

#if (XERCES_VERSION_MAJOR < 3)
    xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    {
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    }
#endif

    xercesc::MemBufFormatTarget *xmlMemTarget = new xercesc::MemBufFormatTarget();

#if (XERCES_VERSION_MAJOR < 3)
    if (!writer->writeNode(xmlMemTarget)
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete xmlMemTarget;
        delete writer;
        return NULL;
    }
#else
    xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    output->setByteStream(xmlMemTarget);

    if (!writer->write(xmlDocToWrite, output))
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete output;
        delete xmlMemTarget;
        delete writer;
        return NULL;
    }

    delete output;
#endif

    delete writer;

    return xmlMemTarget;
}

xercesc::DOMElement *OpenScenarioBase::getRootElement(const std::string &fileName, const std::string &memberName, const std::string &fileType, const bool validate, std::string *errorMessage)
{
    //
    //parsing a file is done in two steps
    // -first step with enabled XInclude and disabled validation
    // -second step with disabled XInclude and enabled validation
    //
    //(if file parsed in one step with enabled XInclude and validation
    // then the validation is done before XInclude. But we want validate the
    // whole document with all elements included in one document
    //

    //parse file with enabled XInclude and disabled validation
    //
    //parser will process XInclude nodes
    parser->setDoXInclude(true);
    //parse without validation
    parser->setDoSchema(false);

    //parse the file
    try
    {
        parser->parse(fileName.c_str());
    }
    catch (...)
    {
        std::cerr << "\nErrors during parse of the document '" << fileName << "'.\n" << std::endl;

        return NULL;
    }


    //validate the xosc file in a second run without XInclude
    // (enabled XInclude generate a namespace attribute xmlns:xml for xml:base
    //  and this had to be added to the xml schema)
    //
    xercesc::DOMElement *tmpRootElem = NULL;
    std::string tmpRootElemName;
    xercesc::DOMDocument *tmpXmlDoc = parser->getDocument();
    if (tmpXmlDoc)
    {
        tmpRootElem = tmpXmlDoc->getDocumentElement();
        tmpRootElemName = xercesc::XMLString::transcode(tmpRootElem->getNodeName());
    }

    //validation
    if (validate)
    {
        //path to covise directory
        bf::path coDir = getEnvVariable("COVISEDIR");
        //relative path from covise directory to OpenSCENARIO directory
        bf::path oscDirRelPath = bf::path("src/OpenCOVER/DrivingSim/OpenScenario");
        //relative path from OpenSCENARIO directory to directory with schema files
        bf::path xsdDirRelPath = bf::path("xml-schema");

        //name of XML Schema
        bf::path xsdFileName;
        FileTypeXsdFileNameMap::const_iterator found = s_fileTypeToXsdFileName.find(tmpRootElemName);
        if (found != s_fileTypeToXsdFileName.end())
        {
            xsdFileName = found->second;
        }
        else
        {
            std::cerr << "Error! Can't determine a XSD Schema file for fileType '" << fileType << "'.\n" << std::endl;

            return NULL;
        }

        //path and filename of XML Schema *.xsd file
        bf::path xsdPathFileName = coDir;
        xsdPathFileName /= oscDirRelPath;
        xsdPathFileName /= xsdDirRelPath;
        xsdPathFileName /= xsdFileName;
		xsdPathFileName.make_preferred();

		//set schema location and load grammar if filename changed
		if (xsdPathFileName != m_xsdPathFileName)
		{
			m_xsdPathFileName = xsdPathFileName;

			//set location of schema for elements without namespace
			// (in schema file no global namespace is used)
			parser->setExternalNoNamespaceSchemaLocation(m_xsdPathFileName.c_str());

			//read the schema grammar file (.xsd) and cache it
			try
			{
				parser->loadGrammar(m_xsdPathFileName.c_str(), xercesc::Grammar::SchemaGrammarType, true);
			}
			catch (...)
			{
			std::cerr << "\nError! Can't load XML Schema '" << m_xsdPathFileName << "'.\n" << std::endl;

			return NULL;
			} 
		}


        //settings for validation
        parser->setDoXInclude(false); //disable XInclude for validation: to prevent generation of a namespace attribute xmlns:xml for xml:base
        parser->setDoSchema(true);

        //write xml document got from parser to memory buffer
        xercesc::MemBufFormatTarget *tmpXmlMemBufFormat = writeFileToMemory(tmpXmlDoc);

        //raw buffer, length and fake id for memory input source
        const XMLByte *tmpRawBuffer = tmpXmlMemBufFormat->getRawBuffer();
        XMLSize_t tmpRawBufferLength = tmpXmlMemBufFormat->getLen();
        const XMLCh *bufId = xercesc::XMLString::transcode("memBuf");

        //new input source from memory for parser
        xercesc::InputSource *tmpInputSrc = new xercesc::MemBufInputSource(tmpRawBuffer, tmpRawBufferLength, bufId);

        //parse memory buffer
        try
        {
            parser->parse(*tmpInputSrc);
        }
		catch (...)
        {
			std::cerr << "\nErrors during validation of the document '" << fileName << "'.\n" << std::endl;
			if (errorMessage)
			{
				ParserErrorHandler *errorHandler = dynamic_cast<ParserErrorHandler *>(parser->getErrorHandler());
				errorMessage->assign(errorHandler->getErrorMessage());
				errorHandler->releaseErrorMessage();
			}

            delete tmpXmlMemBufFormat;
            delete tmpInputSrc;
            return NULL;
        }

        //delete used objects
        delete tmpXmlMemBufFormat;
        delete tmpInputSrc;
    }


    //set xmlDoc and rootElement
    if (tmpXmlDoc)
    {
        xmlDoc = tmpXmlDoc;
        xercesc::DOMElement *rootElement = tmpRootElem;

        //to ensure that the DOM view of a document is the same as if it were saved and re-loaded
        rootElement->normalize();

        return rootElement;
    }
    else
    {
        return NULL;
    }
}

xercesc::DOMElement *OpenScenarioBase::getDefaultXML(const std::string &fileType)
{
	//path to covise directory
	bf::path coDir = getEnvVariable("COVISEDIR");
	//relative path from covise directory to OpenSCENARIO directory
	bf::path oscDirRelPath = bf::path("src/OpenCOVER/DrivingSim/OpenScenario");
	//relative path from OpenSCENARIO directory to directory with schema files
	bf::path xsdDirRelPath = bf::path("xml-default");

	//name of XML Schema
	bf::path xsdFileName;
	FileTypeXsdFileNameMap::const_iterator found = s_defaultFileTypeMap.find(fileType);
	if (found != s_defaultFileTypeMap.end())
	{
		xsdFileName = found->second;
	}
	else
	{
		std::cerr << "Error! Can't determine a XSD Default file for fileType '" << fileType << "'.\n" << std::endl;

		return NULL;
	}

	//path and filename of XML Schema *.xsd file
	bf::path xsdPathFileName = coDir;
	xsdPathFileName /= oscDirRelPath;
	xsdPathFileName /= xsdDirRelPath;
	xsdPathFileName /= xsdFileName;
	xsdPathFileName.make_preferred();


	//parse file with enabled XInclude and disabled validation
    //
    //parser will process XInclude nodes
    parser->setDoXInclude(true);
    //parse without validation
    parser->setDoSchema(false);

    //parse the file
    try
    {
        parser->parse(xsdPathFileName.c_str());
    }
    catch (...)
    {
        std::cerr << "\nErrors during parse of the document '" << xsdPathFileName << "'.\n" << std::endl;

        return NULL;
    }

    xercesc::DOMElement *tmpRootElem = NULL;
    std::string tmpRootElemName;
    xercesc::DOMDocument *tmpXmlDoc = parser->getDocument();
    if (tmpXmlDoc)
    {
        return tmpXmlDoc->getDocumentElement();
    }

	return NULL;
}


//
bool OpenScenarioBase::parseFromXML(xercesc::DOMElement *rootElement)
{
	std::string fileNamePathStr = xercesc::XMLString::transcode(dynamic_cast<xercesc::DOMNode *>(rootElement)->getBaseURI());
	createSource(fileNamePathStr, xercesc::XMLString::transcode(rootElement->getNodeName()));

    return oscObjectBase::parseFromXML(rootElement, source);
}

//
oscSourceFile *OpenScenarioBase::createSource(const std::string &fileName, const std::string &fileType)
{
    source = new oscSourceFile();
	source->setNameAndPath(fileName, fileType, m_pathFromCurrentDirToDoc);

    addToSrcFileVec(source);

	return source;
}




