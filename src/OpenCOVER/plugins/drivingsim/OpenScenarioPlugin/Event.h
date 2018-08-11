#ifndef EVENT_H
#define EVENT_H

#include<string>
#include <vector>
#include <list>
#include <osg/Vec3>
#include <OpenScenario/schema/oscEvent.h>
#include "StoryElement.h"

class Action;
class Condition;
class Sequence;
class Event : public::OpenScenario::oscEvent, public StoryElement
{
public:
    Event();

    std::list<::Action*> actionList;
    void start(Sequence *currentSequence);
    void stop();
    int finishedEntityActions;
    int activeEntites;

    std::string &getName();

    std::list<Condition*> startConditionList;
    void addCondition(Condition* condition);

};

#endif // EVENT_H
