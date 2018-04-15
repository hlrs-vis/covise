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
    std::string roadId;
    int laneId;
    float inits;
    AgentVehicle *agentVehicle;
    osg::Vec3 entityPosition;
    osg::Vec3 directionVector;
    int actionCounter;


    ReferencePosition* refPos;
    ReferencePosition* newRefPos;
    //void updateRefPos();

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
    void setPosition(osg::Vec3 &newPosition);
    void setDirection(osg::Vec3 &newDirection);

    // follow Trajectory attributes
    osg::Vec3 targetPosition;
    osg::Vec3 totaldirectionVector;
    osg::Vec3 newPosition;
    osg::Vec3 referencePosition;

    int visitedVertices;
    float totalDistance;
    float totaldirectionVectorLength;

    // follow Trajectories functions
    void setTrajectoryDirection();
    void setTrajSpeed(float deltat);
    void followTrajectory(Event *event, int verticesCounter);
    void setTrajectoryDirectionOnRoad();

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
