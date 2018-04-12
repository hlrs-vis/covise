#ifndef Condition_H
#define Condition_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscCondition.h>
class Entity;
class Event;
class Maneuver;
class Act;
class Sequence;

class Condition : public OpenScenario::oscCondition
{

private:
	

public:
	Condition();
	~Condition();

    std::string type;

    // time
    float time;

    // distance
    Entity* passiveCar;
    Entity* activeCar;
    float relativeDistance;

    //termination start
    bool rule;

    Maneuver* checkedManeuver;
    Event* checkedEvent;
    Act* checkedAct;
    Sequence* checkedSequence;

};

#endif // Condition_H
