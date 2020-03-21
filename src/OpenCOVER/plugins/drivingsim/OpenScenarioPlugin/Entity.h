#ifndef ENTITY_H
#define ENTITY_H

#include<string>
#include <TrafficSimulation/AgentVehicle.h>
#include <iostream>
#include <math.h>
#include <list>
using namespace std;
using namespace TrafficSimulation;


class Spline;
class ReferencePosition;
class Action;
class Event;
class LaneChange;
class RoadSystem;
namespace OpenScenario
{
	class oscObject;
	class oscVehicle;
}
class Trajectory;
class Entity {

public:
	list<Entity*> entityList;
    std::string name;
    float speed;
    AgentVehicle *agentVehicle;
    osg::Vec3 entityPosition;
    osg::Vec3 directionVector;
	osg::Vec3 lateralVector;
    Trajectory *trajectory;

    ReferencePosition* refPos;
    ReferencePosition* lastRefPos;
    ReferencePosition* newRefPos;
    Entity* refObject;
	//ReferencePosition* laneId; 

    Entity(OpenScenario::oscObject *object);
    ~Entity();
    void setInitEntityPosition(ReferencePosition* init_refPos);
    void moveLongitudinal();
    std::string &getName();
    void setSpeed(float speed_temp);
    void longitudinalSpeedAction(Event *event, double init_targetSpeed, int shape);
	void doLaneChange(LaneChange *lc, Event *event);
	void finishedEntityActionCounter(Event* event);
	void startDoLaneChange(LaneChange *lc);

    float &getSpeed();
    osg::Vec3 getPosition();
    void setDirection(osg::Vec3 &newDirection);

    // follow Trajectory attributes
	int counter;
    int currentVertex;
    float distanceTraveledFromLastVertex;
	float distanceTraveledFromLastStep;
    float segmentLength;
	float destinationDistance;
	float timePassedFromLastStep;
	double distance;
	double tDistance;
	float dsLane;
	float dtLane;
	double timeLc;
	
	//for sinusfunction;
	double a, b, d;


    // follow Trajectories functions
    void startFollowTrajectory(Trajectory *t);
    void followTrajectory(Event *event);
    void setTrajectoryDirection();

	//LaneChange functions
	double getRelativeLcDistance(int value, double targetOffset);
	double getAbsoluteLcDistance(int value, double targetOffset);
	void getDestinationPositionLc(ReferencePosition* relativePos, ReferencePosition* position,double width, double distance);
	double sinusPosition(double s);
	double cubicPosition(double x);
	
    //Longitudinal attributes
    float dt;
    float old_speed;
    float acceleration;
	OpenScenario::oscVehicle *getVehicle() { return vehicle; };

    vehicleUtil::Road *road;

private:

	OpenScenario::oscVehicle * vehicle;
	OpenScenario::oscObject *object;




};

#endif // ENTITY_H
