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
class Event : public::OpenScenario::oscEvent, public StoryElement
{
public:
    Event();

    std::list<::Action*> actionList;
	void stop();
    int finishedEntityActions;
    int activeEntites;

    //conditions

    bool eventCondition;
    bool eventFinished;
    float startTime;
    std::string startConditionType;
    std::string startAfterManeuver;
    std::string passiveCarName;
    std::string activeCarName;
    float relativeDistance;

    std::string &getName();
    void initialize(int numEntites);

    std::list<Condition*> startConditionList;
    void addCondition(Condition* condition);

};

#endif // EVENT_H
