#include "Act.h"
#include "ScenarioManager.h"

Act::Act() :
	oscAct()
{
}

Act::~Act()
{
}

void Act::finishedParsing()
{
	name = oscAct::name.getValue();
}

void Act::initialize(int noe, list<Maneuver*> &maneuverList_temp, list<Entity*> &activeEntityList_temp)
{
	numberOfExecutions = noe;
	maneuverList = maneuverList_temp;
	activeEntityList = activeEntityList_temp;
	executionCounter = 0;
	actCondition = true;
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
