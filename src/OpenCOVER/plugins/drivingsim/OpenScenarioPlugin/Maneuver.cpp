#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>

Maneuver::Maneuver():
	maneuverCondition(false),
	maneuverFinished(false),
	totalDistance(0),
	visitedVertices(0),
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
	int verticesCounter = polylineVertices.size();
	//substract vectors
	directionVector = targetPosition - currentPos;
	float distance = directionVector.length();
	directionVector.normalize();
	//calculate step distance
	float step_distance = speed*opencover::cover->frameDuration();
	if(totalDistance == 0)
	{
		totalDistance = distance;
	}
	//calculate remaining distance
	totalDistance = totalDistance-step_distance;
	//calculate new position
	newPosition = currentPos+(directionVector*step_distance);
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

void Maneuver::changeSpeedOfEntity(Entity *aktivCar, float dt)
{
	float negativeAcceleration = 50;
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
