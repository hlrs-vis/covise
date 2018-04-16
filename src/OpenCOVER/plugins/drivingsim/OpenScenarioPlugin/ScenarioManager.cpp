#include "ScenarioManager.h"
#include <vector>
#include "ReferencePosition.h"
#include "Act.h"
#include "Action.h"
#include "Sequence.h"
#include "Maneuver.h"
#include "Event.h"
#include "Condition.h"
#include "../../../DrivingSim/OpenScenario/schema/oscEntity.h"
#include "OpenScenarioPlugin.h"
#include <OpenScenario/schema/oscPrivate.h>
#include <OpenScenario/OpenScenarioBase.h>
using namespace OpenScenario;

ScenarioManager::ScenarioManager():
	simulationTime(0),
    scenarioCondition(true)
{
}
void ScenarioManager::restart()
{
	simulationTime = 0;
	initializeEntities();
	for (list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
	{
		Act* currentAct = (*act_iter);
		currentAct->stop();
		for (list<Sequence*>::iterator sequence_iter = currentAct->sequenceList.begin(); sequence_iter != currentAct->sequenceList.end(); sequence_iter++)
		{
			Sequence* currentSequence = (*sequence_iter);
			currentSequence->stop();
			for (list<Maneuver*>::iterator maneuver_iter = currentSequence->maneuverList.begin(); maneuver_iter != currentSequence->maneuverList.end(); maneuver_iter++)
			{
				Maneuver* currentManeuver = (*maneuver_iter);
				currentManeuver->stop();
				for (list<Event*>::iterator event_iter = currentManeuver->eventList.begin(); event_iter != currentManeuver->eventList.end(); event_iter++)
				{
					Event* currentEvent = *event_iter;
					currentEvent->stop();
					if (currentEvent != NULL)
					{
						for (list<Entity*>::iterator entity_iter = currentSequence->actorList.begin(); entity_iter != currentSequence->actorList.end(); entity_iter++)
						{
							Entity* currentEntity = (*entity_iter);
							for (list<Action*>::iterator currentAction = currentEvent->actionList.begin(); currentAction != currentEvent->actionList.end(); currentAction++)
							{
								(*currentAction)->stop();
							}
						}

					}
				}
			}
		}
	}
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

void ScenarioManager::initializeEntities()
{

	//get initial position and speed of entities
	for (list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++)
	{
		Entity *currentEntity = (*entity_iter);
		currentEntity->totalDistance = 0;
		currentEntity->visitedVertices = 0;
		currentEntity->dt = 0.0;

		for (oscPrivateArrayMember::iterator it = OpenScenarioPlugin::instance()->osdb->Storyboard->Init->Actions->Private.begin(); it != OpenScenarioPlugin::instance()->osdb->Storyboard->Init->Actions->Private.end(); it++)
		{
			oscPrivate* actions_private = ((oscPrivate*)(*it));
			if (currentEntity->getName() == actions_private->object.getValue())
			{
				for (oscPrivateActionArrayMember::iterator it2 = actions_private->Action.begin(); it2 != actions_private->Action.end(); it2++)
				{
					oscPrivateAction* action = ((oscPrivateAction*)(*it2));
					if (action->Longitudinal.exists())
					{
						currentEntity->setSpeed(action->Longitudinal->Speed->Target->Absolute->value.getValue());
					}
					if (action->Position.exists())
					{
						Position* initPos = (Position*)(action->Position.getObject());
						if (initPos->Lane.exists())
						{
							ReferencePosition* refPos = new ReferencePosition();
							refPos->init(initPos->Lane->roadId.getValue(), initPos->Lane->laneId.getValue(), initPos->Lane->s.getValue(), OpenScenarioPlugin::instance()->getRoadSystem());
							currentEntity->setInitEntityPosition(refPos);
							currentEntity->refPos = refPos;
						}
						else if (initPos->World.exists())
						{
							osg::Vec3 initPosition = initPos->getAbsoluteWorld();
							double hdg = initPos->getHdg();

							ReferencePosition* refPos = new ReferencePosition();
							refPos->init(initPosition, hdg, OpenScenarioPlugin::instance()->getRoadSystem());
							currentEntity->setInitEntityPosition(refPos);
							currentEntity->refPos = refPos;
						}
						else if (initPos->Road.exists())
						{
							ReferencePosition* refPos = new ReferencePosition();
							refPos->init(initPos->Road->roadId.getValue(), initPos->Road->s.getValue(), initPos->Road->t.getValue(), OpenScenarioPlugin::instance()->getRoadSystem());
							currentEntity->setInitEntityPosition(refPos);
							currentEntity->refPos = refPos;
						}
					}
				}

			}
		}

	}
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
        if(currentAct->isRunning())
        {
            for(std::list<Condition*>::iterator condition_iter = currentAct->endConditionList.begin(); condition_iter != currentAct->endConditionList.end(); condition_iter++)
            {
                Condition* actEndCondition = (*condition_iter);
                if(conditionControl(actEndCondition))
                {
					currentAct->stop();
					fprintf(stderr, "stopping act %s", currentAct->name.c_str());
                }
                else
                {
                    eventConditionManager(currentAct);
                }
            }
        }
        else if(currentAct->isStopped())
        {
            for(std::list<Condition*>::iterator condition_iter = currentAct->startConditionList.begin(); condition_iter != currentAct->startConditionList.end(); condition_iter++)
            {
                Condition* actStartCondition = (*condition_iter);
                if(conditionControl((actStartCondition)))
                {
                    currentAct->start();
					fprintf(stderr, "starting act %s", currentAct->name.c_str());
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
                if(currentEvent->isStopped())
                {
                    for(std::list<Condition*>::iterator condition_iter = currentEvent->startConditionList.begin(); condition_iter != currentEvent->startConditionList.end(); condition_iter++)
                    {
                        Condition* eventCondition = (*condition_iter);

                        if(conditionControl(eventCondition))
                        {
							currentEvent->initialize(currentSequence->actorList.size());
                            currentEvent->start();
							fprintf(stderr, "starting event %s\n", currentEvent->name.getValue().c_str());
                            currentManeuver->activeEvent = currentEvent;
                            break;
                        }
                    }
                }
                if(currentEvent->isRunning())
                {
                    if(currentEvent->activeEntites*currentEvent->actionList.size() == currentEvent->finishedEntityActions)
                    {
                        // lieber running actions und dann abziehen
                        currentEvent->finish();
						fprintf(stderr, "finishing event %s\n", currentEvent->name.getValue().c_str());
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
                currentManeuver->finish();

				fprintf(stderr, "finishing maneuver %s", currentManeuver->name.getValue().c_str());
                finishedManeuvers++;
            }
            else if(currentManeuver->activeEvent == NULL)
            {
                currentManeuver->stop();
            }
            else
            {
                currentSequence->activeManeuver = currentManeuver;
                break;
            }
        }
        if(finishedManeuvers == currentSequence->maneuverList.size())
        {
            currentSequence->finish();
            finishedSequences++;
        }
        else if(currentSequence->activeManeuver == NULL)
        {
            currentSequence->stop();
        }
    }
    if(finishedSequences == currentAct->sequenceList.size())
    {
        currentAct->finish();
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
            if(condition->checkedManeuver->isFinished())
            {
                condition->set(true);
                condition->waitForDelay = true;
                return condition->delayReached();
            }
        }
        else if(condition->checkedEvent != NULL)
        {
            if(condition->checkedEvent->isFinished())
            {
                condition->set(true);
                condition->waitForDelay = true;
                return condition->delayReached();
            }
        }
    }
    if(condition->ByEntity.exists())
    {
        Entity* passiveEntity = condition->passiveEntity;
        float relativeDistance = condition->ByEntity->EntityCondition->RelativeDistance->value.getValue();
        for(std::list<Entity*>::iterator entity_iter = condition->activeEntityList.begin(); entity_iter != condition->activeEntityList.end(); entity_iter++)
        {
            Entity* activeEntity = (*entity_iter);
            if((activeEntity->refPos->roadId == passiveEntity->refPos->roadId) && (activeEntity->refPos->s- passiveEntity->refPos->s >= relativeDistance)) // todo entities might be on different roads
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

void ScenarioManager::resetReferencePositionStatus()
{
    for (list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++)
    {
        Entity* currentEntity = (*entity_iter);
        currentEntity->refPos->resetStatus();
    }
}
