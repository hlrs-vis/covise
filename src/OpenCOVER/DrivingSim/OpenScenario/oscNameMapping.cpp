/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#include "oscNameMapping.h"
using namespace OpenScenario;

nameMapping::nameMapping()
{
	bm.insert(bm_type::value_type(parent_name("OpenSCENARIO", "/Catalog"), "CatalogOpenSCENARIO"));
	bm.insert(bm_type::value_type(parent_name("Catalog", "/Catalog/CatalogOpenSCENARIO"), "CatalogObject"));
	bm.insert(bm_type::value_type(parent_name("Position", "/OSCPosition/PositionRoute"), "RoutePosition"));
	bm.insert(bm_type::value_type(parent_name("ByEntity", "/OSCCondition/ByEntity/EntityCondition/Collision"), "CollisionByEntity"));
	bm.insert(bm_type::value_type(parent_name("ByEntity", "/OpenSCENARIO/Entities/Selection/Members"), "MembersByEntity"));
	bm.insert(bm_type::value_type(parent_name("ConditionGroup", "/OSCManeuver/Event/Conditions/Start"), "StartConditionGroup"));
	bm.insert(bm_type::value_type(parent_name("Target", "/OSCCondition/ByEntity/EntityCondition/TimeToCollision"), "CollisionTarget"));
	bm.insert(bm_type::value_type(parent_name("Target", "Speed"), "SpeedTarget"));
	bm.insert(bm_type::value_type(parent_name("Target", "/OSCPrivateAction/Lateral/LaneChange"), "LaneChangeTarget"));
	bm.insert(bm_type::value_type(parent_name("Target", "/OSCPrivateAction/Lateral/LaneOffset"), "LaneOffsetTarget"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "entryName"), "EntryParameter"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "/OSCCondition/ByValue"), "ConditionParameter"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "/OSCGlobalAction"), "ActionParameter"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "/OSCParameterAssignment"), "SetParameter"));
	bm.insert(bm_type::value_type(parent_name("Waypoint", "/OSCTrajectory"), "Vertex"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "execution"), "ExecutionParameter"));
	bm.insert(bm_type::value_type(parent_name("Route", "/OSCPosition"), "PositionRoute"));
	bm.insert(bm_type::value_type(parent_name("Longitudinal", "/OSCPrivateAction/Routing/FollowTrajectory"), "LongitudinalParams"));
	bm.insert(bm_type::value_type(parent_name("Lateral", "/OSCPrivateAction/Routing/FollowTrajectory"), "LateralParams"));
	bm.insert(bm_type::value_type(parent_name("Signal", "/OSCCondition/ByState"), "SignalState"));
	bm.insert(bm_type::value_type(parent_name("Description", "OSCPedestrianController"), "Description"));
	bm.insert(bm_type::value_type(parent_name("Distance", "/OSCPrivateAction/Longitudinal"), "DistanceAction"));
	bm.insert(bm_type::value_type(parent_name("Dynamics", "/OSCPrivateAction/Longitudinal/Speed"), "SpeedDynamics"));
	bm.insert(bm_type::value_type(parent_name("Dynamics", "/OSCPrivateAction/Lateral/LaneChange"), "LaneChangeDynamics"));
	bm.insert(bm_type::value_type(parent_name("Dynamics", "/OSCPrivateAction/Lateral/LaneOffset"), "LaneOffsetDynamics"));
	bm.insert(bm_type::value_type(parent_name("Entity", "/OSCGlobalAction"), "ActionEntity"));
	bm.insert(bm_type::value_type(parent_name("Add", "/OSCGlobalAction/ActionParameter/Modify/Rule"), "AddRule"));
	bm.insert(bm_type::value_type(parent_name("TimeOfDay", "/OSCEnvironment"), "EnvironmentTimeOfDay"));
	bm.insert(bm_type::value_type(parent_name("Speed", "/OSCCondition/ByEntity/EntityCondition"), "ConditionSpeed"));
	bm.insert(bm_type::value_type(parent_name("Controller", "/OSCPrivateAction"), "ActionController"));
	bm.insert(bm_type::value_type(parent_name("Controller", "/OpenSCENARIO/RoadNetwork/Signals"), "SignalsController"));
	bm.insert(bm_type::value_type(parent_name("Controller", "/OpenSCENARIO/Entities/Object"), "ObjectController"));
	bm.insert(bm_type::value_type(parent_name("Conditions", "/OpenSCENARIO/Storyboard/Story/Act"), "ActConditions"));
	bm.insert(bm_type::value_type(parent_name("Distance", "/OSCCondition/ByEntity/EntityCondition"), "ConditionDistance"));
	bm.insert(bm_type::value_type(parent_name("OpenSCENARIO", "/OpenSCENARIO/Storyboard/End"), "OpenSCENARIOEnd"));
	bm.insert(bm_type::value_type(parent_name("Relative", "/OSCPrivateAction/Longitudinal/Speed/Target"), "RelativeTarget"));
	bm.insert(bm_type::value_type(parent_name("Catalog", "/OpenSCENARIO"), "CatalogObject"));
	//bm.insert(bm_type::value_type(parent_name("Signal", "/OpenSCENARIO/RoadNetwork/Signals/Controller/Phase"), "SignalPhase"));
	//bm.insert(bm_type::value_type(parent_name("Signal", "/OpenSCENARIO/RoadNetwork/Signals/SignalsController/Phase"), "SignalPhase"));
	bm.insert(bm_type::value_type(parent_name("Signal", "/OSCGlobalAction/Infrastructure"), "InfrastructureSignal"));

	//bm.insert(bm_type::value_type(parent_name("Relative", "/OSCPrivateAction/Lateral/LaneChange/Target"), "LaneChangeTargetRelative"));
	//bm.insert(bm_type::value_type(parent_name("Relative", "/OSCPrivateAction/Lateral/LaneOffset/Target"), "LaneOffsetTargetRelative"));
	//bm.insert(bm_type::value_type(parent_name("Relative", "/OSCPrivateAction/Lateral/LaneChange/LaneChangeTarget"), "LaneChangeTargetRelative"));
	//bm.insert(bm_type::value_type(parent_name("Relative", "/OSCPrivateAction/Lateral/LaneOffset/LaneOffsetTarget"), "LaneOffsetTargetRelative"));
	bm.insert(bm_type::value_type(parent_name("Position", "/OSCPosition/Route"), "RoutePosition"));
	
	
	
};
nameMapping *nameMapping::nmInstance=NULL;
nameMapping *nameMapping::instance()
{
	if (nmInstance == NULL)
	{
		nmInstance = new nameMapping();
	}
	return nmInstance;
}

std::string nameMapping::getClassName(const std::string &name, std::string parent)
{
	auto search = bm.left.find(parent_name(name,parent));
	if (search != bm.left.end()) {
		return (*search).second;
	}
	else {
		return name;
	}
}
std::string nameMapping::getSchemaName(std::string &className)
{
	auto search = bm.right.find(className);
	if (search != bm.right.end()) {
		return (*search).first;
	}
	else {
		return className;
	}
}
