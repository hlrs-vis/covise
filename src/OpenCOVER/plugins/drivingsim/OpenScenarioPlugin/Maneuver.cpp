#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>

Maneuver::Maneuver(string name):
	name(name),
	maneuverCondition(true),
	totalDistance(0),
	visitedVertices(0),
	verticesCounter(0)
{
}
Maneuver::~Maneuver()
{
}

osg::Vec3 &Maneuver::followTrajectory(osg::Vec3 currentPos, osg::Vec3 targetPosition, float speed)
{
	//substract vectors
	osg::Vec3 targetPos(targetPosition[0],targetPosition[1],targetPosition[2]);
	normDirectionVec = targetPos - currentPos;
	float distance = normDirectionVec.length();
	normDirectionVec.normalize();
	//calculate step distance
	float step_distance = 0.1*speed*opencover::cover->frameDuration();//speed
	if(totalDistance==0){totalDistance=distance;}
	//calculate remaining distance
	totalDistance=totalDistance-step_distance;
	//calculate new position
	newPosition = currentPos+(normDirectionVec*step_distance);
	if (totalDistance<0)
	{
		visitedVertices++;
		totalDistance=0;
		if (visitedVertices==verticesCounter)
		{
			maneuverCondition=false;
		}
	}
	return newPosition;
}
string Maneuver::getName()
{
	return name;
}

bool Maneuver::getManeuverCondition()
{
return maneuverCondition;
}
void Maneuver::setManeuverCondition()
{
	if ( maneuverCondition==true)
	{ 
		maneuverCondition=false;
	}
	if ( maneuverCondition==false)
	{ 
		maneuverCondition=true;
	}
}

void Maneuver::setPolylineVertices(osg::Vec3 polyVec)
{
	polylineVertices.push_back(polyVec);
	verticesCounter++;
}
