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

class oscByEntity;
class Condition : public OpenScenario::oscCondition
{

private:
	

public:
	Condition();
	~Condition();

    bool isTrue;

    // (longitudinal) distance
    Entity* passiveCar;
    Entity* activeCar;

    //termination start
    Maneuver* checkedManeuver;
    Event* checkedEvent;
    Act* checkedAct;
    Sequence* checkedSequence;

    void initalize(oscByEntity* condition);

};

#endif // Condition_H
