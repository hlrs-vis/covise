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

class Sequence : public::OpenScenario::oscSequence, public StoryElement
{
public:
    Sequence();
	virtual void stop();
    std::list<::Maneuver*> maneuverList;
    std::list<Entity*> actorList;
    int executions;
    std::string name;
    ::Maneuver* activeManeuver;


    void initialize(std::list<Entity*> actorList_temp, std::list<::Maneuver *> maneuverList_temp);
};

#endif // SEQUENCE_H
