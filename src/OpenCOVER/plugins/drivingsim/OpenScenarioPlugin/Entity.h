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
    std::string roadId;
    int laneId;
    float inits;
    AgentVehicle *entityGeometry;
    osg::Vec3 entityPosition;
    osg::Vec3 directionVector;
    int actionCounter;


    ReferencePosition* refPos;
    ReferencePosition* newRefPos;
    //void updateRefPos();

    Entity(std::string entityName, std::string catalogReferenceName);
    ~Entity();
    void setInitEntityPosition(osg::Vec3 init);
    void setInitEntityPosition(Road *r);
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




};

#endif // ENTITY_H
