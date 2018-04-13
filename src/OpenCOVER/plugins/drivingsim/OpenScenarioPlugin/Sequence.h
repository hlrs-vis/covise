#ifndef SEQUENCE_H
#define SEQUENCE_H

#include<string>
#include <vector>
#include <list>
#include <osg/Vec3>
#include <OpenScenario/schema/oscSequence.h>
#include "StoryElement.h"

class Maneuver;
class Entity;
class Event;
class Sequence : public::OpenScenario::oscSequence, public StoryElement
{
public:
    Sequence();
    std::list<::Maneuver*> maneuverList;
    std::list<Entity*> actorList;
    int executions;
    std::string name;
    Event* activeEvent;


    void initialize(std::list<Entity*> actorList_temp, std::list<::Maneuver *> maneuverList_temp);
};

#endif // SEQUENCE_H
