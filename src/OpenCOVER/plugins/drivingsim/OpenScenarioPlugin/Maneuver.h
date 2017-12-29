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
    std::string trajectoryCatalogReference;

    float deltat = 0.1;
    //conditions
    bool maneuverCondition;
    bool maneuverFinished;
    float startTime;
    std::string startConditionType;
    std::string startAfterManeuver;
    std::string passiveCar;
    std::string activeCar;
    float relativeDistance;
    float targetSpeed;

    //followTrajectory
    float totalDistance;
    osg::Vec3 directionVector;
    osg::Vec3 totaldirectionVector;
    osg::Vec3 verticeStartPos;
    float totaldirectionVectorLength;
    std::list<Trajectory*> trajectoryList;
    int visitedVertices;
    bool arriveAtVertex;
    osg::Vec3 targetPosition;
    osg::Vec3 newPosition;

    Maneuver();
    ~Maneuver();
    virtual void finishedParsing();
    std::string &getName();
    osg::Vec3 &followTrajectory(osg::Vec3 currentPos, std::vector<osg::Vec3> polylineVertices, float speed, float timer);
    osg::Vec3 &followTrajectoryRel(osg::Vec3 currentPos, std::vector<osg::Vec3> polylineVertices, float speed);
    void changeSpeedOfEntity(Entity *aktivCar, float dt);
};

#endif // MANEUVER_H
