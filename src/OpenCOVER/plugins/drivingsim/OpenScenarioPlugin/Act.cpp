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
	//executionCounter = 0;
	actCondition = false;
	actFinished = false;
	endTime = 0;
}

int Act::getNumberOfExecutions()
{
	return numberOfExecutions;
}

string Act::getName()
{
	return name;
}


