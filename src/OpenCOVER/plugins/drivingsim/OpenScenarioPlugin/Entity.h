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
class Entity {

public:
    std::string name;
    std::string catalogReferenceName;
    std::string filepath;
    float speed;
    AgentVehicle *entityGeometry;
    osg::Vec3 entityPosition;
    osg::Vec3 directionVector;

    ReferencePosition* refPos;
    ReferencePosition* newRefPos;
    Entity* refObject;

    Entity(std::string entityName, std::string catalogReferenceName);
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




};

#endif // ENTITY_H
