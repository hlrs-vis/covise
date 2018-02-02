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

    std::list<Trajectory*> trajectoryList;

    Maneuver();
    ~Maneuver();

    void checkConditions();

    virtual void finishedParsing();

    std::string &getName();
    void changeSpeedOfEntity(Entity *aktivCar, float dt);

};

#endif // MANEUVER_H
