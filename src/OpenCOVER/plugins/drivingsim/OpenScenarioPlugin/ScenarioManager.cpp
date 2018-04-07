#include "ScenarioManager.h"
#include <vector>
#include "ReferencePosition.h"
#include "Act.h"
#include "Sequence.h"
#include "Maneuver.h"
#include "Event.h"

ScenarioManager::ScenarioManager():
	simulationTime(0),
    scenarioCondition(true),
    anyActTrue(false)
{
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

void ScenarioManager::conditionManager(){
    if(conditionControl()){
        for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
        {
            if(conditionControl((*act_iter)))
            {
                for(list<Sequence*>::iterator sequence_iter = (*act_iter)->sequenceList.begin(); sequence_iter != (*act_iter)->sequenceList.end(); sequence_iter++)
                {
                    anyActTrue = true;
                    bool anyEvent = false;
                    for(list<Maneuver*>::iterator maneuver_iter = (*sequence_iter)->maneuverList.begin(); maneuver_iter != (*sequence_iter)->maneuverList.end(); maneuver_iter++)
                    {
                        for(list<Event*>::iterator event_iter = (*maneuver_iter)->eventList.begin(); event_iter != (*maneuver_iter)->eventList.end(); event_iter++)
                        {
                            if(conditionControl((*event_iter),(*maneuver_iter)))
                            {
                                (*sequence_iter)->activeEvent = (*event_iter);
                                anyEvent = true;
                            }
                            if(!anyEvent)
                            {
                                (*sequence_iter)->activeEvent = NULL;
                            }
                        }
                    }
                }
            }
        }
    }
}

bool ScenarioManager::conditionControl()
{
	if (endTime<simulationTime)
		{
			scenarioCondition = false;
            return false;
		}
    return true;
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
    if ((event->finishedEntityActions) == (event->activeEntites*event->actionVector.size()))
    {
        for(list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++)
        {
            (*entity_iter)->resetActionAttributes();
        }
        event->eventFinished = true;
        event->eventCondition = false;
        maneuver->finishedEvents = maneuver->finishedEvents+1;
        if(maneuver->finishedEvents == maneuver->eventList.size())
        {
            maneuver->maneuverFinished = true;
        }

        return event->eventCondition;
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




