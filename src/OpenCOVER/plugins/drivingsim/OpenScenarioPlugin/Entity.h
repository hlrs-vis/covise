#ifndef ENTITY_H
#define ENTITY_H

#include<string>
#include <TrafficSimulation/AgentVehicle.h>
#include <iostream>
#include <math.h>

//#include "Spline.h"
class Spline;
class ReferencePosition;
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
    void longitudinalSpeedAction(std::list<Entity*> *activeEntityList, double init_targetSpeed, int shape);

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
    void followTrajectoryOnRoad(int verticesCounter,std::list<Entity*> *activeEntityList);
    void setTrajectoryDirectionOnRoad();

    //Longitudinal attributes
    float dt;
    float old_speed;
    float acceleration;




};

#endif // ENTITY_H
