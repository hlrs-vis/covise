#include "Act.h"
#include "ScenarioManager.h"

Act::Act(string actName, int numberOfExecutions, list<Maneuver*> maneuverList_temp, list<Entity*> activeEntityList_temp):
	name(actName), 
	numberOfExecutions(numberOfExecutions), 
	maneuverList(maneuverList_temp), 
	activeEntityList(activeEntityList_temp),
	executionCounter(0),
	actCondition(true)
{
}

Act::~Act()
{
}

int Act::getNumberOfExecutions()
{
	return numberOfExecutions;
}

int Act::getExecutionCounter()
{
	return executionCounter;
}

void Act::setExecutionCounter()
{
	executionCounter++;
}

string Act::getName()
{
	return name;
}

bool Act::getActCondition()
{
	return actCondition;
}

void Act::setActCondition()
{
	if (actCondition == true)
	{
		actCondition = false;
	}
	if (actCondition == false)
	{
		actCondition = true;
	}
}
