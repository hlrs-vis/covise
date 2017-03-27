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

 private:
	string name;


 public:
	Maneuver();
	~Maneuver();
	virtual void finishedParsing();

	float startTime;
	string startConditionType;
	string maneuverType;
	string startAfterManeuver;

	string passiveCar;
	string activeCar;
	float relativeDistance;

	float targetSpeed;

	string trajectoryCatalogReference;
	float totalDistance;
	osg::Vec3 normDirectionVec;
	list<Trajectory*> trajectoryList;
	//vector<osg::Vec3> polylineVertices;
	int visitedVertices;
	int verticesCounter;
	osg::Vec3 newPosition;
	osg::Vec3 targetPosition;
	bool maneuverCondition;
	bool arriveAtVertex;
    osg::Vec3 &followTrajectory(osg::Vec3 currentPos, vector<osg::Vec3> polylineVertices, float speed);
	string &getName();
	bool getManeuverCondition();
	void setManeuverCondition();
	//void setPolylineVertices(osg::Vec3 polyVec);
	void simulationTimeConditionControl(float simulationTime);
	void distanceToEntityConditionControl(Entity *aktivCar, Entity *passiveCar);
	void maneuverTerminitionControl(Maneuver *terminatedManeuver);
	void changeSpeedOfEntity(Entity *aktivCar, float dt);
	bool maneuverFinished;
};

#endif // MANEUVER_H
