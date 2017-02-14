/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#include "oscNameMapping.h"
using namespace OpenScenario;

nameMapping::nameMapping()
{
	bm.insert(bm_type::value_type(parent_name("Position", "Route"), "RoutePosition"));
	bm.insert(bm_type::value_type(parent_name("ByEntity", "Collision"), "CollisionByEntity"));
	bm.insert(bm_type::value_type(parent_name("ConditionGroup", "Start"), "StartConditionGroup"));
	bm.insert(bm_type::value_type(parent_name("Target", "TimeOfCollision"), "CollisionTarget"));
	bm.insert(bm_type::value_type(parent_name("Target", "Speed"), "SpeedTarget"));
	bm.insert(bm_type::value_type(parent_name("Target", "LaneChange"), "LaneChangeTarget"));
	bm.insert(bm_type::value_type(parent_name("Target", "LaneOffset"), "LaneOffsetTarget"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "entryName"), "EntryParameter"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "ByValue"), "ConditionParameter"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "OSCGlobalAction"), "ActionParameter"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "OSCParameterAssignment"), "SetParameter"));
	bm.insert(bm_type::value_type(parent_name("Waypoint", "OSCTrajectory"), "Vertex"));
	bm.insert(bm_type::value_type(parent_name("Parameter", "execution"), "ExecutionParameter"));
	bm.insert(bm_type::value_type(parent_name("Route", "OSCPosition"), "PositionRoute"));
	bm.insert(bm_type::value_type(parent_name("Longitudinal", "FollowTrajectory"), "LongitudinalParams"));
	bm.insert(bm_type::value_type(parent_name("Lateral", "FollowTrajectory"), "LateralParams"));
	bm.insert(bm_type::value_type(parent_name("Signal", "ByState"), "SignalState"));
	bm.insert(bm_type::value_type(parent_name("Description", "OSCPedestrianController"), "Description"));
	bm.insert(bm_type::value_type(parent_name("Distance", "Longitudinal"), "DistanceAction"));
	bm.insert(bm_type::value_type(parent_name("Description", "OSCPedestrianController"), "Description"));
	bm.insert(bm_type::value_type(parent_name("Dynamics", "Speed"), "SpeedDynamics"));
	bm.insert(bm_type::value_type(parent_name("Dynamics", "LaneChange"), "LaneChangeDynamics"));
	bm.insert(bm_type::value_type(parent_name("Dynamics", "LaneOffset"), "LaneOffsetDynamics"));
	bm.insert(bm_type::value_type(parent_name("Entity", "OSCGlobalAction"), "ActionEntity"));
	bm.insert(bm_type::value_type(parent_name("Add", "Rule"), "AddRule"));
	bm.insert(bm_type::value_type(parent_name("TimeOfDay", "OSCEnvironment"), "EnvironmentTimeOfDay"));
	bm.insert(bm_type::value_type(parent_name("Speed", "EntityCondition"), "ConditionSpeed"));
	bm.insert(bm_type::value_type(parent_name("Controller", "OSCPrivateAction"), "ActionController"));
	bm.insert(bm_type::value_type(parent_name("Distance", "EntityCondition"), "ConditionDistance"));
	bm.insert(bm_type::value_type(parent_name("Relative", "LaneChangeTarget"), "LaneChangeTargetRelative"));
	bm.insert(bm_type::value_type(parent_name("Relative", "LaneOffsetTarget"), "LaneOffsetTargetRelative"));
	bm.insert(bm_type::value_type(parent_name("Position", "PositionRoute"), "RoutePosition"));
	
	
	
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

std::string nameMapping::getClassName(std::string &name, std::string parent)
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
