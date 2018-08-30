#ifndef Condition_H
#define Condition_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscCondition.h>
#include<list>

class Entity;
class Event;
class Maneuver;
class Act;
class Sequence;
class ScenarioManager;

class Condition : public OpenScenario::oscCondition
{

private:
	

public:
	Condition();
	~Condition();

    bool isTrue;
    float delayTimer;
    bool waitForDelay;
    float distance;
    // (longitudinal) distance
    Entity* passiveEntity;
    std::list<Entity*> activeEntityList;

    //termination start
    Maneuver* checkedManeuver;
    Event* checkedEvent;
    Act* checkedAct;
    Sequence* checkedSequence;

    void addActiveEntity(Entity* entity);
    void setPassiveEntity(Entity* entity);
    void setManeuver(Maneuver* maneuver);
    void setEvent(Event* event);
    void set(bool state);
    void increaseTimer();
    bool delayReached();
};

#endif // Condition_H
