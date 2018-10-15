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
#include "Event.h"
#include "Sequence.h"
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
    if (name == "oscEvent")
    {
        return new Event();
    }
    if (name == "oscSequence")
    {
        return new Sequence();
    }
	return OpenScenario::oscFactories::staticObjectFactory.create(name);
}
