#ifndef ENTITY_H
#define ENTITY_H

#include<string>
#include <TrafficSimulation/AgentVehicle.h>
#include <iostream>
#include <math.h>

class Spline;
class ReferencePosition;
class Action;
class Event;
namespace OpenScenario
{
	class oscObject;
	class oscVehicle;
}
class Trajectory;
class Entity {

public:
    std::string name;
    float speed;
    AgentVehicle *agentVehicle;
    osg::Vec3 entityPosition;
    osg::Vec3 directionVector;
    Trajectory *trajectory;

    ReferencePosition* refPos;
    ReferencePosition* lastRefPos;
    ReferencePosition* newRefPos;
    Entity* refObject;

    Entity(OpenScenario::oscObject *object);
    ~Entity();
    void setInitEntityPosition(ReferencePosition* init_refPos);
    void moveLongitudinal();
    std::string &getName();
    void setSpeed(float speed_temp);
    void longitudinalSpeedAction(Event *event, double init_targetSpeed, int shape);

    float &getSpeed();
    osg::Vec3 getPosition();
    void setDirection(osg::Vec3 &newDirection);

    // follow Trajectory attributes

    int currentVertex;
    float distanceTraveledFromLastVertex;
    float segmentLength;

    // follow Trajectories functions
    void startFollowTrajectory(Trajectory *t);
    void setTrajSpeed(float deltat);
    void followTrajectory(Event *event);
    void setTrajectoryDirection();

    //Longitudinal attributes
    float dt;
    float old_speed;
    float acceleration;
	OpenScenario::oscVehicle *getVehicle() { return vehicle; };

private:

	OpenScenario::oscVehicle * vehicle;
	OpenScenario::oscObject *object;




};

#endif // ENTITY_H
