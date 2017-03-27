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

/*int Act::getExecutionCounter()
{
	return executionCounter;
}

void Act::setExecutionCounter()
{
	executionCounter++;
}*/

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

void Act::simulationTimeConditionControl(float simulationTime)
{
	if(startTime<simulationTime && actFinished==false)
	{
		actCondition = true;
	}
	else
	{
		actCondition = false;
	}

	if (endTime != 0)
	{
		if(startTime<simulationTime && endTime>simulationTime && actFinished==false)
		{
			actCondition = true;
		}
		else
		{
			actCondition = false;
		}
	}
}

Maneuver* Act::getManeuverByName(string maneuverName)
{
	for (list<Maneuver*>::iterator maneuver_iter = maneuverList.begin(); maneuver_iter != maneuverList.end(); maneuver_iter++)
	{
		if ((*maneuver_iter)->getName() != maneuverName)
		{
			continue;
			return 0;
		}
		else
		{
			return (*maneuver_iter);
		}
	}
return 0;
}

