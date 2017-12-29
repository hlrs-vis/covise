#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>

using namespace std;

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

osg::Vec3 &Maneuver::followTrajectory(osg::Vec3 currentPos, vector<osg::Vec3> polylineVertices, float speed,float timer)
{
    int verticesCounter = polylineVertices.size();

    if(totalDistance == 0)
    {

        verticeStartPos = currentPos;
        targetPosition = polylineVertices[visitedVertices];

        float xcoord = currentPos[0];

        fprintf(stderr,"%f, ",timer);
        fprintf(stderr, "%s, %i, %f, %f,\n",name.c_str(),visitedVertices,xcoord,targetPosition[0]);

        if (visitedVertices > 170){
            int blup = 1;
        }
    }
//    std::string testvar = name.c_str();

//    if (testvar == "laneChange0") {
//        fprintf(stderr, " x[%i], %f, %f,\n",visitedVertices,timer,currentPos[0]);
//    }


    //substract vectors
    directionVector = targetPosition - currentPos;
    float distance = directionVector.length();


    // calculate speed
    totaldirectionVector = targetPosition - verticeStartPos;
    totaldirectionVectorLength = totaldirectionVector.length();
    speed = totaldirectionVectorLength/deltat;

    directionVector.normalize();

    //calculate step distance
    float step_distance = speed*opencover::cover->frameDuration();
    //float step_distance = speed*1/60;

    if(totalDistance <= 0)
	{
		totalDistance = distance;
	}
	//calculate remaining distance
	totalDistance = totalDistance-step_distance;
	//calculate new position
	newPosition = currentPos+(directionVector*step_distance);
    if (totalDistance <= 0)
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

osg::Vec3 &Maneuver::followTrajectoryRel(osg::Vec3 currentPos, vector<osg::Vec3> polylineVertices, float speed)
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
