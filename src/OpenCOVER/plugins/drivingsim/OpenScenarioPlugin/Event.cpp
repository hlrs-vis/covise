#include "Event.h"
#include "Condition.h"
#include "StoryElement.h"

Event::Event():
    StoryElement(),
    finishedEntityActions(0)
{

}

void Event::stop()
{
	finishedEntityActions=0;
	activeEntites=0;
	StoryElement::stop();
}

void Event::initialize(int numEntites)
{
    activeEntites = numEntites;
}

void Event::addCondition(Condition *condition)
{
    startConditionList.push_back(condition);

}
