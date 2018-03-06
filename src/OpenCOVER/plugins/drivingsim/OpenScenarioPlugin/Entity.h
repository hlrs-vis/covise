#ifndef ENTITY_H
#define ENTITY_H

#include<string>
#include <TrafficSimulation/AgentVehicle.h>
#include <iostream>

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

    // Splines
    std::string activeShape;
    Spline *spline;
    void followSpline();
    float splineDistance;
    int visitedSplineVertices;
    void setActiveShape(std::string);
    osg::Vec3 splinePos;
    void setSplinePos(osg::Vec3);

    ReferencePosition* refPos;

    Entity(std::string entityName, std::string catalogReferenceName);
    ~Entity();
    void setInitEntityPosition(osg::Vec3 init);
    void setInitEntityPosition(osg::Vec3 init, osg::Vec3 initDirVec);
    void setInitEntityPosition(osg::Matrix m);
    void setInitEntityPosition(Road *r);
    void moveLongitudinal();
    std::string &getName();
    void setSpeed(float speed_temp);

    float &getSpeed();
    osg::Vec3 &getPosition();
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
    bool refPosIsSet;
    bool finishedCurrentTraj;


    // follow Trajectories functions
    void setTrajectoryDirection(osg::Vec3 init_targetPosition);
    void followTrajectory(int verticesCounter, std::list<Entity*> &finishedEntityList);
    void getTrajSpeed(float deltat);
    void setRefPos();
    void setRefPos(osg::Vec3 newReferencePosition);


};

#endif // ENTITY_H
