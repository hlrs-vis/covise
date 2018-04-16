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
class Entity {

public:
    std::string name;
    float speed;
    AgentVehicle *agentVehicle;
    osg::Vec3 entityPosition;
    osg::Vec3 directionVector;

    ReferencePosition* refPos;
    ReferencePosition* newRefPos;
    Entity* refObject;

    Entity(OpenScenario::oscObject *object);
    ~Entity();
    void setInitEntityPosition(ReferencePosition* init_refPos);
    void moveLongitudinal();
    std::string &getName();
    void setSpeed(float speed_temp);
    void longitudinalSpeedAction(Event *event, double init_targetSpeed, int shape);
    void resetActionAttributes();

    float &getSpeed();
    osg::Vec3 getPosition();
    void setDirection(osg::Vec3 &newDirection);

    // follow Trajectory attributes
    osg::Vec3 targetPosition;
    osg::Vec3 totaldirectionVector;

    int visitedVertices;
    float totalDistance;
    float totaldirectionVectorLength;

    // follow Trajectories functions
    void setTrajSpeed(float deltat);
    void followTrajectory(Event *event, int verticesCounter);
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
