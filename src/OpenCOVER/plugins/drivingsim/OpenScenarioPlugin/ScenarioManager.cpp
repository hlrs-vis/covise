#include "ScenarioManager.h"
#include <vector>
#include "ReferencePosition.h"
#include "Act.h"
#include "Sequence.h"
#include "Maneuver.h"
#include "Event.h"
#include "Condition.h"
#include "../../../DrivingSim/OpenScenario/schema/oscEntity.h"

ScenarioManager::ScenarioManager():
	simulationTime(0),
    scenarioCondition(true),
    anyActTrue(false)
{
}

Entity* ScenarioManager::getEntityByName(std::string entityName)
{
	for (list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++)
	{
		if ((*entity_iter)->getName() != entityName)
		{
			continue;
			return 0;
		}
		else
		{
			return (*entity_iter);
		}
	}
return 0;
}

Maneuver* ScenarioManager::getManeuverByName(std::string maneuverName)
{
    for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
    {
        for(list<Sequence*>::iterator sequence_iter = (*act_iter)->sequenceList.begin(); sequence_iter != (*act_iter)->sequenceList.end(); sequence_iter++)
        {
            for(list<Maneuver*>::iterator maneuver_iter = (*sequence_iter)->maneuverList.begin(); maneuver_iter != (*sequence_iter)->maneuverList.end(); maneuver_iter++)
            {
                Maneuver* maneuver = (*maneuver_iter);
                if(maneuver->name.getValue() == maneuverName)
                {
                    return maneuver;
                }
            }
        }
    }
    return NULL;
}
Event* ScenarioManager::getEventByName(std::string eventName)
{
    for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
    {
        for(list<Sequence*>::iterator sequence_iter = (*act_iter)->sequenceList.begin(); sequence_iter != (*act_iter)->sequenceList.end(); sequence_iter++)
        {
            for(list<Maneuver*>::iterator maneuver_iter = (*sequence_iter)->maneuverList.begin(); maneuver_iter != (*sequence_iter)->maneuverList.end(); maneuver_iter++)
            {
                for(list<Event*>::iterator event_iter = (*maneuver_iter)->eventList.begin(); event_iter != (*maneuver_iter)->eventList.end(); event_iter++)
                {
                    Event* event = (*event_iter);
                    if(event->name.getValue() == eventName)
                    {
                        return event;
                    }
                }
            }
        }
    }
    return NULL;
}

void ScenarioManager::conditionManager(){
    // check Story end condition
    for(std::list<Condition*>::iterator condition_iter = endConditionList.begin(); condition_iter != endConditionList.end(); condition_iter++)
    {
        Condition* storyCondition = (*condition_iter);
        if(conditionControl(storyCondition)) // if endcondition is false, story is true
        {
            scenarioCondition = false;
        }
        else
        {
            scenarioCondition = true;
            actConditionManager();
        }
    }
}
void ScenarioManager::actConditionManager()
{
    //check Act conditions
    for(std::list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
    {
        Act* currentAct = (*act_iter);
        // endconditions
        if(currentAct->StoryElement::isRunning())
        {
            for(std::list<Condition*>::iterator condition_iter = currentAct->endConditionList.begin(); condition_iter != currentAct->endConditionList.end(); condition_iter++)
            {
                Condition* actEndCondition = (*condition_iter);
                if(conditionControl(actEndCondition))
                {
                    currentAct->StoryElement::stop();
                }
                else
                {
                    eventConditionManager(currentAct);
                }
            }
        }
        else if(currentAct->StoryElement::isStopped())
        {
            for(std::list<Condition*>::iterator condition_iter = currentAct->startConditionList.begin(); condition_iter != currentAct->startConditionList.end(); condition_iter++)
            {
                Condition* actStartCondition = (*condition_iter);
                if(conditionControl((actStartCondition)))
                {
                    currentAct->StoryElement::start();
                    eventConditionManager(currentAct);
                }
            }
        }
    }
}

void ScenarioManager::eventConditionManager(Act* currentAct)
{
    int finishedSequences = 0;
    for(std::list<Sequence*>::iterator sequence_iter = currentAct->sequenceList.begin(); sequence_iter != currentAct->sequenceList.end(); sequence_iter++)
    {
        int finishedManeuvers = 0;
        Sequence* currentSequence = (*sequence_iter);
        currentSequence->activeManeuver = NULL;
        currentSequence->start();
        for(std::list<Maneuver*>::iterator maneuver_iter = currentSequence->maneuverList.begin(); maneuver_iter != currentSequence->maneuverList.end(); maneuver_iter++)
        {
            int finishedEvents = 0;
            Maneuver* currentManeuver = (*maneuver_iter);
            currentManeuver->activeEvent = NULL;
            currentManeuver->start();
            for(std::list<Event*>::iterator event_iter = currentManeuver->eventList.begin(); event_iter != currentManeuver->eventList.end(); event_iter++)
            {
                Event* currentEvent = (*event_iter);
                if(currentEvent->StoryElement::isStopped())
                {
                    for(std::list<Condition*>::iterator condition_iter = currentEvent->startConditionList.begin(); condition_iter != currentEvent->startConditionList.end(); condition_iter++)
                    {
                        Condition* eventCondition = (*condition_iter);

                        if(conditionControl(eventCondition))
                        {
                            currentEvent->StoryElement::start();
                            currentManeuver->activeEvent = currentEvent;
                            break;
                        }
                    }
                }
                if(currentEvent->StoryElement::isRunning())
                {
                    if(currentEvent->activeEntites*currentEvent->actionList.size() == currentEvent->finishedEntityActions)
                    {
                        // lieber running actions und dann abziehen
                        currentEvent->StoryElement::finish();
                        finishedEvents++;
                    }
                    else
                    {
                        if(currentManeuver->activeEvent == NULL)
                        {
                            currentManeuver->activeEvent = currentEvent;
                        }
                    }
                }
            }
            if(finishedEvents == currentManeuver->eventList.size())
            {
                currentManeuver->StoryElement::finish();
                finishedManeuvers++;
            }
            else if(currentManeuver->activeEvent == NULL)
            {
                currentManeuver->StoryElement::stop();
            }
            else
            {
                currentSequence->activeManeuver = currentManeuver;
                break;
            }
        }
        if(finishedManeuvers == currentSequence->maneuverList.size())
        {
            currentSequence->StoryElement::finish();
            finishedSequences++;
        }
        else if(currentSequence->activeManeuver == NULL)
        {
            currentSequence->StoryElement::stop();
        }
    }
    if(finishedSequences == currentAct->sequenceList.size())
    {
        currentAct->StoryElement::finish();
    }
}

bool ScenarioManager::conditionControl(Condition* condition)
{
    if(condition->waitForDelay)
    {
        return condition->delayReached();
    }
    if(condition->ByValue.exists())
    {
        if (condition->ByValue->SimulationTime->value.getValue()<simulationTime)
        {
            condition->set(true);
            condition->waitForDelay = true;
            return condition->delayReached();
        }
    }
    if(condition->ByState.exists())
    {
        if(condition->checkedManeuver != NULL)
        {
            if(condition->checkedManeuver->StoryElement::state == StoryElement::finished)
            {
                condition->set(true);
                condition->waitForDelay = true;
                return condition->delayReached();
            }
        }
        else if(condition->checkedEvent != NULL)
        {
            if(condition->checkedEvent->StoryElement::state == StoryElement::finished)
            {
                condition->set(true);
                condition->waitForDelay = true;
                return condition->delayReached();
            }
        }
    }
    if(condition->ByEntity.exists())
    {
        Entity* pasiveEntity = condition->passiveEntity;
        float relativeDistance = condition->ByEntity->EntityCondition->RelativeDistance->value.getValue();
        for(std::list<Entity*>::iterator entity_iter = condition->activeEntityList.begin(); entity_iter != condition->activeEntityList.end(); entity_iter++)
        {
            Entity* activeEntity = (*entity_iter);
            if (activeEntity->refPos->s-pasiveEntity->refPos->s >= relativeDistance)
            {
                condition->set(true);
                condition->waitForDelay = true;
                return condition->delayReached();
            }
        }
    }

    return false;
}

void ScenarioManager::addCondition(Condition *condition)
{
    endConditionList.push_back(condition);
}

void ScenarioManager::initializeCondition(Condition *condition)
{
    if(condition->ByValue.exists())
    {
        // nothing to initialize. everything is alsready stored in oscCondition
    }
    if(condition->ByState.exists())
    {
        int type = condition->ByState->AfterTermination->type.getValue();
        if(type == 2) // if type is a maneuver
        {
            std::string maneuverName = condition->ByState->AfterTermination->name.getValue();
            condition->setManeuver(getManeuverByName(maneuverName));
        }
        else if(type == 3) // if type is an event
        {
            std::string eventName = condition->ByState->AfterTermination->name.getValue();
            condition->setEvent(getEventByName(eventName));
        }
    }
    if(condition->ByEntity.exists())
    {
        std::string passiveEntityName = condition->ByEntity->EntityCondition->RelativeDistance->entity.getValue();
        condition->setPassiveEntity(getEntityByName(passiveEntityName));

        for (oscEntityArrayMember::iterator it = condition->ByEntity->TriggeringEntities->Entity.begin(); it != condition->ByEntity->TriggeringEntities->Entity.end(); it++)
        {
            oscEntity* entity = ((oscEntity*)(*it));
            std::string activeEntityName = entity->name.getValue();
            condition->addActiveEntity(getEntityByName(activeEntityName));
        }
    }
}
