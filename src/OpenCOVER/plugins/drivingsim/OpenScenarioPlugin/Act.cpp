#include "Act.h"
#include "ScenarioManager.h"
#include "Condition.h"
#include "StoryElement.h"

Act::Act() :
    oscAct(),
    StoryElement()
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
    sequenceList.push_back(sequence_temp);
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
