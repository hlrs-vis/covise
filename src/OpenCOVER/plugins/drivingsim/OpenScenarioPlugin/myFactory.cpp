#include "myFactory.h"
#include "Maneuver.h"
#include "Act.h"
#include "Trajectory.h"
#include "Condition.h"
#include "RouteInfo.h"
#include "FollowTrajectory.h"
#include "Position.h"
#include "Action.h"
#include "StartConditions.h"
#include "EndConditions.h"
#include <DrivingSim/OpenScenario/oscFactories.h>

myFactory::myFactory(){}
myFactory::~myFactory(){}

OpenScenario::oscObjectBase *myFactory::create(const std::string &name)
{
	if (name == "oscManeuver")
	{
		return new Maneuver();
	}
	if (name == "oscAct")
	{
		return new Act();
	}
	if (name == "oscRouteInfo")
	{
		return new RouteInfo();
	}
	if (name == "oscTrajectory")
	{
		return new Trajectory();
	}
	if (name == "oscCondition")
	{
		return new Condition();
	}
	if (name == "oscFollowTrajectory")
	{
		return new FollowTrajectory();
	}
    if (name == "oscPosition")
    {
        return new Position();
    }
    if (name == "oscAction")
    {
        return new Action();
    }
    if (name == "oscStartConditions")
    {
        return new StartConditions();
    }
    if (name == "oscEndConditions")
    {
        return new EndConditions();
    }
	return OpenScenario::oscFactories::staticObjectFactory.create(name);
}
