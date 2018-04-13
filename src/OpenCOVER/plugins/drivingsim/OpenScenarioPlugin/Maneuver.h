#ifndef MANEUVER_H
#define MANEUVER_H

#include<string>
#include "Trajectory.h"
#include "Entity.h"
#include <vector>
#include <list>
#include <osg/Vec3>
#include <OpenScenario/schema/oscManeuver.h>
class Action;
class Event;
class Maneuver: public OpenScenario::oscManeuver
{


public:
    std::string maneuverName;
    std::string maneuverType;
    std::string routeCatalogReference;
    std::string trajectoryCatalogReference;
    //conditions
    int finishedEvents;
    bool maneuverCondition;
    bool maneuverFinished;
    float startTime;
    std::string startConditionType;
    std::string startAfterManeuver;
    std::string passiveCarName;
    std::string activeCarName;
    int numberOfActiveEntities;
    float relativeDistance;
    float targetSpeed;

    std::list<::Event*> eventList;

    Maneuver();
    ~Maneuver();

    void checkConditions();

    virtual void finishedParsing();

    std::string &getName();
    void changeSpeedOfEntity(Entity *aktivCar, float dt, std::list<Entity *> *activeEntityList);
    void initialize(::Event* event_temp);

};

#endif // MANEUVER_H
