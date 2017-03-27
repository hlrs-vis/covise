#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>

Maneuver::Maneuver():
	maneuverCondition(false),
	maneuverFinished(false),
	totalDistance(0),
	visitedVertices(0),
	verticesCounter(0),
	trajectoryCatalogReference(""),
	startAfterManeuver(""),
	startConditionType("termination"),
	targetSpeed(0)
{
}
Maneuver::~Maneuver()
{
}

void Maneuver::finishedParsing()
{
	name = oscManeuver::name.getValue();
}

osg::Vec3 &Maneuver::followTrajectory(osg::Vec3 currentPos, vector<osg::Vec3> polylineVertices, float speed)
{
	if(totalDistance == 0)
	{
	targetPosition = currentPos + polylineVertices[visitedVertices];
	}
	verticesCounter = polylineVertices.size();
	//substract vectors
	normDirectionVec = targetPosition - currentPos;
	float distance = normDirectionVec.length();
	normDirectionVec.normalize();
	//calculate step distance
	float step_distance = 0.1*speed*opencover::cover->frameDuration();//speed
	if(totalDistance == 0)
	{
		totalDistance=distance;
	}
	//calculate remaining distance
	totalDistance=totalDistance-step_distance;
	//calculate new position
	newPosition = currentPos+(normDirectionVec*step_distance);
	if (totalDistance < 0)
	{
		visitedVertices++;
		totalDistance = 0;
		if (visitedVertices == verticesCounter)
		{
			maneuverCondition = false;
			maneuverFinished = true;
		}
	}
	return newPosition;
}
string &Maneuver::getName()
{
	return name;
}

bool Maneuver::getManeuverCondition()
{
return maneuverCondition;
}

void Maneuver::setManeuverCondition()
{
	if ( maneuverCondition == true)
	{ 
		maneuverCondition = false;
	}
	if ( maneuverCondition == false)
	{ 
		maneuverCondition = true;
	}
}

/*void Maneuver::setPolylineVertices(osg::Vec3 polyVec)
{
	polylineVertices.push_back(polyVec);
	verticesCounter++;
}*/

void Maneuver::simulationTimeConditionControl(float simulationTime)
{
	if(startTime<simulationTime && maneuverFinished != true)
	{
		maneuverCondition = true;
	}
	else
	{
		maneuverCondition = false;
	}
}

void Maneuver::distanceToEntityConditionControl(Entity *aktivCar, Entity *passiveCar)
{
if (aktivCar->entityPosition[0]-passiveCar->entityPosition[0] >= relativeDistance && maneuverFinished == false)
	{
		maneuverCondition = true;
	}
}

void Maneuver::maneuverTerminitionControl(Maneuver *terminatedManeuver)
{
if (terminatedManeuver->maneuverFinished == true && maneuverFinished == false && terminatedManeuver->getName() == startAfterManeuver)
	{
		maneuverCondition = true;
	}
}

void Maneuver::changeSpeedOfEntity(Entity *aktivCar, float dt)
{
	float negativeAcceleration = 70;
	float dv = negativeAcceleration*dt;
	if(aktivCar->getSpeed()>targetSpeed)
	{
		aktivCar->setSpeed(aktivCar->getSpeed()-dv);
	}
	else
	{
	aktivCar->setSpeed(targetSpeed);
	}
}
