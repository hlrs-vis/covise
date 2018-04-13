#include "Condition.h"
#include "Entity.h"

using namespace std;

Condition::Condition():
    oscCondition(),
    isTrue(false),
    delayTimer(0.0),
    passiveEntity(NULL),
    checkedAct(NULL),
    checkedEvent(NULL),
    checkedManeuver(NULL),
    checkedSequence(NULL)
{}
Condition::~Condition(){}

void Condition::addActiveEntity(Entity* entity)
{
    activeEntityList.push_back(entity);
}

void Condition::setPassiveEntity(Entity* entity)
{
    passiveEntity = entity;
}

void Condition::setManeuver(Maneuver* maneuver)
{
    checkedManeuver = maneuver;
}

void Condition::set(bool state)
{
    isTrue = state;
}
