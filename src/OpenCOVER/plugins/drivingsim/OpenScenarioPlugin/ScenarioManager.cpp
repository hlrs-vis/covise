#include "ScenarioManager.h"
#include <vector>
#include "ReferencePosition.h"
#include "Act.h"
#include "Sequence.h"
#include "Maneuver.h"
#include "Event.h"
#include "Condition.h"
#include "OpenScenarioPlugin.h"
#include <OpenScenario/schema/oscPrivate.h>
#include <OpenScenario/OpenScenarioBase.h>
using namespace OpenScenario;

ScenarioManager::ScenarioManager():
	simulationTime(0),
    scenarioCondition(true),
    anyActTrue(false)
{
}
void ScenarioManager::restart()
{
	simulationTime = 0;
	initializeEntities();
}

Entity* ScenarioManager::getEntityByName(string entityName)
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

Maneuver* ScenarioManager::getManeuverByName(string maneuverName)
{
    for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
    {

        for(list<Sequence*>::iterator sequence_iter = (*act_iter)->sequenceList.begin(); sequence_iter != (*act_iter)->sequenceList.end(); sequence_iter++)
        {
            for(list<Maneuver*>::iterator maneuver_iter = (*sequence_iter)->maneuverList.begin(); maneuver_iter != (*sequence_iter)->maneuverList.end(); maneuver_iter++)
            {
                if((*maneuver_iter)->name.getValue() == maneuverName)
                {
                    Maneuver* maneuver = (*maneuver_iter);
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
		Entity *currentTentity = (*entity_iter);

		for (oscPrivateArrayMember::iterator it = OpenScenarioPlugin::instance()->osdb->Storyboard->Init->Actions->Private.begin(); it != OpenScenarioPlugin::instance()->osdb->Storyboard->Init->Actions->Private.end(); it++)
		{
			oscPrivate* actions_private = ((oscPrivate*)(*it));
			if (currentTentity->getName() == actions_private->object.getValue())
			{
				for (oscPrivateActionArrayMember::iterator it2 = actions_private->Action.begin(); it2 != actions_private->Action.end(); it2++)
				{
					oscPrivateAction* action = ((oscPrivateAction*)(*it2));
					if (action->Longitudinal.exists())
					{
						currentTentity->setSpeed(action->Longitudinal->Speed->Target->Absolute->value.getValue());
					}
					if (action->Position.exists())
					{
						Position* initPos = (Position*)(action->Position.getObject());
						if (initPos->Lane.exists())
						{
							ReferencePosition* refPos = new ReferencePosition();
							refPos->init(initPos->Lane->roadId.getValue(), initPos->Lane->laneId.getValue(), initPos->Lane->s.getValue(), OpenScenarioPlugin::instance()->getRoadSystem());
							currentTentity->setInitEntityPosition(refPos);
							currentTentity->refPos = refPos;
						}
						else if (initPos->World.exists())
						{
							osg::Vec3 initPosition = initPos->getAbsoluteWorld();
							double hdg = initPos->getHdg();

							ReferencePosition* refPos = new ReferencePosition();
							refPos->init(initPosition, hdg, OpenScenarioPlugin::instance()->getRoadSystem());
							currentTentity->setInitEntityPosition(refPos);
							currentTentity->refPos = refPos;
						}
						else if (initPos->Road.exists())
						{
							ReferencePosition* refPos = new ReferencePosition();
							refPos->init(initPos->Road->roadId.getValue(), initPos->Road->s.getValue(), initPos->Road->t.getValue(), OpenScenarioPlugin::instance()->getRoadSystem());
							currentTentity->setInitEntityPosition(refPos);
							currentTentity->refPos = refPos;
						}
					}
				}

			}
		}

	}
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
            //check Act conditions
            for(std::list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
            {
                Act* currentAct = (*act_iter);
                // endconditions
                for(std::list<Condition*>::iterator condition_iter = currentAct->endConditionList.begin(); condition_iter != currentAct->endConditionList.end(); condition_iter++)
                {
                    Condition* actEndCondition = (*condition_iter);
                    if(conditionControl(actEndCondition))
                    {
                        currentAct->actCondition = false;
                        continue;
                    }
                }
                for(std::list<Condition*>::iterator condition_iter = currentAct->startConditionList.begin(); condition_iter != currentAct->startConditionList.end(); condition_iter++)
                {
                    Condition* actStartCondition = (*condition_iter);
                    if(conditionControl((actStartCondition)))
                    {
                        currentAct->actCondition = true;
                        for(std::list<Sequence*>::iterator sequence_iter = (*act_iter)->sequenceList.begin(); sequence_iter != (*act_iter)->sequenceList.end(); sequence_iter++)
                        {
                            Sequence* currentSequence = (*sequence_iter);
                            for(std::list<Maneuver*>::iterator maneuver_iter = currentSequence->maneuverList.begin(); maneuver_iter != currentSequence->maneuverList.end(); maneuver_iter++)
                            {
                                Maneuver* currentManeuver = (*maneuver_iter);
                                for(std::list<Event*>::iterator event_iter = currentManeuver->eventList.begin(); event_iter != currentManeuver->eventList.end(); event_iter++)
                                {
                                    Event* currentEvent = (*event_iter);
                                    for(std::list<Condition*>::iterator condition_iter = currentEvent->startConditionList.begin(); condition_iter != currentEvent->startConditionList.end(); condition_iter++)
                                    {
                                        Condition* eventCondition = (*condition_iter);
                                        if(conditionControl(eventCondition))
                                        {
                                            if(currentEvent->activeEntites*currentEvent->actionList.size() == currentEvent->finishedEntityActions)
                                            {
                                                currentEvent->eventFinished = true;
                                                currentManeuver->maneuverFinished = true;
                                            }
                                            else
                                            {
                                                currentEvent->eventCondition = true;
                                                currentSequence->activeEvent = currentEvent;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

bool ScenarioManager::conditionControl(Condition* condition)
{
    if(condition->isTrue)
    {
        return true;
    }
    if(condition->ByValue.exists())
    {
        if (condition->ByValue->SimulationTime->value.getValue()<simulationTime)
        {
            condition->set(true);
            return true;
        }
    }
    if(condition->ByState.exists())
    {
        if(condition->checkedManeuver != NULL)
        {
            if(condition->checkedManeuver->maneuverFinished == true)
            {
                condition->set(true);
                return true;
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
                return true;
            }
        }
    }

    return false;
}

bool ScenarioManager::conditionControl(Act* act)
{
	if (act->startConditionType=="time")
	{
		if(act->startTime<simulationTime && act->actFinished==false)
		{
            act->actCondition = true;
            return act->actCondition;
		}
		else
		{
			act->actCondition = false;
            return act->actCondition;
		}

		if (endTime != 0)
		{
			if(act->startTime<simulationTime && endTime>simulationTime && act->actFinished==false)
			{
				act->actCondition = true;
                return act->actCondition;
			}
			else
			{
				act->actCondition = false;
                return act->actCondition;
			}
		}
	}
    return false;
}

bool ScenarioManager::conditionControl(Event* event,Maneuver* maneuver)
{
    if(event->eventFinished)
    {
        return false;
    }
    if (event->startConditionType=="time")
	{
        if(event->startTime<simulationTime && event->eventFinished != true)
		{
            event->eventCondition = true;
            return event->eventCondition;
		}
		else
		{
            event->eventCondition = false;
            return event->eventCondition;
		}
	}
    if (event->startConditionType=="distance")
	{
        auto activeCar = getEntityByName(event->activeCarName);
        auto passiveCar = getEntityByName(event->passiveCarName);
        if (activeCar->refPos->s-passiveCar->refPos->s >= event->relativeDistance && event->eventFinished == false)
		{
            event->eventCondition = true;
            return event->eventCondition;
		}

    }
    if (event->startConditionType=="termination")
    {
        for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
        {
            for(list<Sequence*>::iterator sequence_iter = (*act_iter)->sequenceList.begin(); sequence_iter != (*act_iter)->sequenceList.end(); sequence_iter++)
            {
                for(list<Maneuver*>::iterator terminatedManeuver = (*sequence_iter)->maneuverList.begin(); terminatedManeuver != (*sequence_iter)->maneuverList.end(); terminatedManeuver++)
                {
                    if ((*terminatedManeuver)->maneuverFinished == true && maneuver->maneuverFinished == false && (*terminatedManeuver)->getName() == event->startAfterManeuver)
                    {
                        event->eventCondition = true;
                        return event->eventCondition;
                    }
                }
            }
        }
    }
    return false;
}

void ScenarioManager::addCondition(Condition *condition)
{
    endConditionList.push_back(condition);
}



