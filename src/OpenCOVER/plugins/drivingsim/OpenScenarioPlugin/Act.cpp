#include "Act.h"
#include "ScenarioManager.h"
#include "Condition.h"

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

void Act::initialize(::Sequence *sequence_temp)
{
    numberOfExecutions = 1;
    sequenceList.push_back(sequence_temp);
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

void Act::addEndCondition(Condition *condition)
{
    endConditionList.push_back(condition);

}

void Act::addStartCondition(Condition *condition)
{
    startConditionList.push_back(condition);

}
