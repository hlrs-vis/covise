#include "Condition.h"
#include "Entity.h"
#include "OpenScenarioPlugin.h"

using namespace std;

Condition::Condition():
    oscCondition(),
    isTrue(false),
    waitForDelay(false),
    delayTimer(0.0),
    passiveEntity(NULL),
    checkedAct(NULL),
    checkedEvent(NULL),
    checkedManeuver(NULL),
    checkedSequence(NULL)
{
    distance = -10000000;
}
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

void Condition::setEvent(Event* event)
{
    checkedEvent = event;
}
void Condition::set(bool state)
{
    isTrue = state;
}

void Condition::increaseTimer()
{
    delayTimer = delayTimer + OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
}

bool Condition::delayReached()
{
    if(delayTimer >= delay.getValue())
    {
        delayTimer = 0.0;
        waitForDelay = false;
        return true;
    }
    increaseTimer();
    return false;
}
