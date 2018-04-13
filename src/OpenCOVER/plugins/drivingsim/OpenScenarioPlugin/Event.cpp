#include "Event.h"
#include "Condition.h"

Event::Event():
    finishedEntityActions(0),
    eventFinished(false),
    eventCondition(false)
{

}

void Event::initialize(int numEntites)
{
    activeEntites = numEntites;
}

void Event::addCondition(Condition *condition)
{
    startConditionList.push_back(condition);

}
