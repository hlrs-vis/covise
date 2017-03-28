#ifndef MANEUVER_H
#define MANEUVER_H

using namespace std;
#include<iostream>
#include<string>
#include<Trajectory.h>
#include<Entity.h>
#include <vector>
#include <list>
#include <algorithm>
#include <osg/Vec3>
#include <DrivingSim/OpenScenario/schema/oscManeuver.h>


class Maneuver: public OpenScenario::oscManeuver
{

 public:
	string name;
	string maneuverType;
	string trajectoryCatalogReference;

	//conditions
	bool maneuverCondition;
	bool maneuverFinished;
	float startTime;
	string startConditionType;
	string startAfterManeuver;
	string passiveCar;
	string activeCar;
	float relativeDistance;
	float targetSpeed;

	//followTrajectory
	float totalDistance;
	osg::Vec3 directionVector;
	list<Trajectory*> trajectoryList;
	int visitedVertices;
	bool arriveAtVertex;
	osg::Vec3 targetPosition;
	osg::Vec3 newPosition;

	Maneuver();
	~Maneuver();
	virtual void finishedParsing();
	string &getName();
    osg::Vec3 &followTrajectory(osg::Vec3 currentPos, vector<osg::Vec3> polylineVertices, float speed);
	void changeSpeedOfEntity(Entity *aktivCar, float dt);
};

#endif // MANEUVER_H
