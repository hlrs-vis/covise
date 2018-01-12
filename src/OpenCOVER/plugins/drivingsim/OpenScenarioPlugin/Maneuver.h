#ifndef MANEUVER_H
#define MANEUVER_H

#include<string>
#include "Trajectory.h"
#include "Entity.h"
#include <vector>
#include <list>
#include <osg/Vec3>
#include <OpenScenario/schema/oscManeuver.h>


class Maneuver: public OpenScenario::oscManeuver
{


public:
    std::string name;
    std::string maneuverType;
    std::string routeCatalogReference;
    std::string trajectoryCatalogReference;
    float deltat = 0.1;
    //conditions
    bool maneuverCondition;
    bool maneuverFinished;
    float startTime;
    std::string startConditionType;
    std::string startAfterManeuver;
    std::string passiveCarName;
    std::string activeCarName;
    float relativeDistance;
    float targetSpeed;

    //followTrajectory
    float totalDistance;
    float speed;
    osg::Vec3 totaldirectionVector;
    osg::Vec3 verticeStartPos;
    float totaldirectionVectorLength;
    int verticesCounter;
    osg::Vec3 directionVector;

    std::list<Trajectory*> trajectoryList;
    int visitedVertices;
    bool arriveAtVertex;
    osg::Vec3 targetPosition;
    osg::Vec3 newPosition;

    Maneuver();
    ~Maneuver();

    void checkConditions();

    virtual void finishedParsing();

    std::string &getName();
    osg::Vec3 &followTrajectory(osg::Vec3 currentPos, std::vector<osg::Vec3> polylineVertices, float timer);
    osg::Vec3 &followTrajectoryRel(osg::Vec3 currentPos, osg::Vec3 targetPosition, float speed);
    void changeSpeedOfEntity(Entity *aktivCar, float dt);
    float &getTrajSpeed(osg::Vec3 verticeStartPos, osg::Vec3 polylineVertices);
    osg::Vec3 &checkRelVertex(osg::Vec3 currentPos, std::vector<osg::Vec3> polylineVertices,std::vector<bool> isRelVertice);
    void setTargetPosition(osg::Vec3 init_TargetPosition);
};

#endif // MANEUVER_H
